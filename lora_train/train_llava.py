from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from joint_data import (
    apply_cli_overrides,
    build_parser,
    compute_metrics,
    load_config,
    load_experiment_records,
    make_run_dir,
    resolve_record_image,
    run_metadata,
    save_json,
    save_predictions,
    trainer_log_history,
)
from model_utils import (
    collect_lora_update_stats,
    get_lora_target_manifest,
    make_quantization_config,
    prepare_lora_model,
)

MODEL_KEY = "llava"
DEFAULT_CONFIG = "config/llava.yaml"


# ============================================================
# Collator
# ============================================================

class LlavaCollator:
    def __init__(
        self,
        processor: Any,
        dataset_root: Path,
        max_length: int,
    ):
        self.processor = processor
        self.dataset_root = dataset_root
        self.max_length = max_length

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:

        texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []

        for record in features:
            image_path = resolve_record_image(
                self.dataset_root,
                record,
            )

            images.append(
                Image.open(image_path).convert("RGB")
            )

            prompt = (
                f"USER: <image>\n"
                f"{record['prompt']} "
                f"ASSISTANT:"
            )

            answer = str(
                record["answer"]
            )

            eos = (
                self.processor.tokenizer.eos_token
                or ""
            )

            prompt_texts.append(
                prompt
            )

            texts.append(
                f"{prompt} {answer}{eos}"
            )

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch[
            "input_ids"
        ].clone()

        prompt_lengths = (
            prompt_batch[
                "attention_mask"
            ]
            .sum(dim=1)
            .tolist()
        )

        for index, prompt_length in enumerate(
            prompt_lengths
        ):
            labels[
                index,
                :int(prompt_length),
            ] = -100

            labels[
                index,
                batch["attention_mask"][index] == 0,
            ] = -100

        batch["labels"] = labels

        return batch


# ============================================================
# Generation evaluation
# ============================================================

def generate_predictions(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    dataset_root: Path,
    config: dict[str, Any],
) -> list[str]:

    import torch

    was_training = model.training

    model.eval()

    device = next(
        model.parameters()
    ).device

    generation_cfg = (
        config.get(
            "evaluation",
            {},
        )
        or {}
    )

    max_new_tokens = int(
        generation_cfg.get(
            "max_new_tokens",
            config.get(
                "training",
                {},
            ).get(
                "max_new_tokens",
                64,
            ),
        )
    )

    predictions: list[str] = []

    for record in records:
        image_path = resolve_record_image(
            dataset_root,
            record,
        )

        image = (
            Image.open(
                image_path
            )
            .convert("RGB")
        )

        text = (
            f"USER: <image>\n"
            f"{record['prompt']} "
            f"ASSISTANT:"
        )

        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(device)
                if hasattr(
                    value,
                    "to",
                )
                else value
            )
            for key, value
            in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = (
            inputs[
                "input_ids"
            ]
            .shape[1]
        )

        generated = output_ids[
            :,
            prompt_len:
        ]

        prediction = (
            processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        predictions.append(
            prediction
        )

    if was_training:
        model.train()

    return predictions


# ============================================================
# Metric helpers
# ============================================================

def _metric_value(
    block: dict[str, Any],
) -> float:
    """
    兼容：
      value_percent
      value

    VQA / ScienceQA 通常是 value_percent。
    Flickr30k CIDEr 可能是 value。
    """

    if "value_percent" in block:
        return float(
            block["value_percent"]
        )

    if "value" in block:
        return float(
            block["value"]
        )

    return 0.0


def checkpoint_selection_key(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    """
    AND checkpoint 选择规则。

    优先级：

    1. 是否通过 strict AND criterion
    2. strict_and_score
    3. and_gap
    4. ASR_B
    5. -ASR_T
    6. -ASR_I
    7. clean performance

    Python tuple 按词典序比较。
    """

    clean = metrics[
        "clean_performance"
    ]

    attack = metrics[
        "attack_success_rate"
    ]

    leakage = metrics[
        "single_trigger_target_leakage"
    ]

    and_quality = metrics[
        "and_quality"
    ]

    cp = _metric_value(
        clean
    )

    asr_b = _metric_value(
        attack
    )

    asr_i = _metric_value(
        leakage[
            "asr_img_only"
        ]
    )

    asr_t = _metric_value(
        leakage[
            "asr_txt_only"
        ]
    )

    strict_score = float(
        and_quality.get(
            "strict_and_score_percent",
            0.0,
        )
    )

    and_gap = float(
        and_quality.get(
            "and_gap_percent",
            asr_b
            - max(
                asr_i,
                asr_t,
            ),
        )
    )

    passes = bool(
        and_quality.get(
            "passes_recommended_criterion",
            False,
        )
    )

    return (
        float(int(passes)),
        strict_score,
        and_gap,
        asr_b,
        -asr_t,
        -asr_i,
        cp,
    )


# ============================================================
# AND checkpoint selector
# ============================================================

class AndCheckpointSelector:
    """
    Trainer callback。

    每次 Trainer 保存 checkpoint 时：

        1. 对 CP / B / I / T 做真实 generation；
        2. compute_metrics；
        3. 根据 strict-AND 规则比较；
        4. 如果当前最好，则保存到 best_adapter/。

    best_adapter 独立于 checkpoints/，
    不会被 save_total_limit 删除。
    """

    def __init__(
        self,
        processor: Any,
        experiment_data: dict[str, Any],
        config: dict[str, Any],
        run_dir: Path,
        enabled: bool = True,
    ):
        self.processor = processor
        self.experiment_data = experiment_data
        self.config = config
        self.run_dir = run_dir
        self.enabled = enabled

        self.best_key: tuple[
            float,
            ...
        ] | None = None

        self.best_step: str | int | None = None

        self.best_metrics: (
            dict[str, Any]
            | None
        ) = None

        self.eval_dir = (
            run_dir
            / "checkpoint_eval"
        )

        self.eval_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def evaluate_current_model(
        self,
        model: Any,
        step: str | int,
        save_split_predictions: bool = False,
    ) -> dict[str, Any]:

        predictions: dict[
            str,
            list[str],
        ] = {}

        for (
            split,
            records,
        ) in self.experiment_data[
            "eval"
        ].items():

            split_predictions = (
                generate_predictions(
                    model,
                    self.processor,
                    records,
                    self.experiment_data[
                        "dataset_root"
                    ],
                    self.config,
                )
            )

            predictions[
                split
            ] = split_predictions

            if save_split_predictions:
                pred_dir = (
                    self.eval_dir
                    / f"step_{step}"
                )

                pred_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_predictions(
                    pred_dir
                    / f"{split}_predictions.jsonl",
                    split,
                    records,
                    split_predictions,
                    self.experiment_data[
                        "target"
                    ],
                )

        metrics = compute_metrics(
            self.experiment_data[
                "eval"
            ],
            predictions,
            self.experiment_data[
                "target"
            ],
        )

        save_json(
            self.eval_dir
            / f"step_{step}.json",
            metrics,
        )

        return metrics

    def maybe_update_best(
        self,
        model: Any,
        metrics: dict[str, Any],
        step: str | int,
    ) -> bool:

        current_key = (
            checkpoint_selection_key(
                metrics
            )
        )

        is_best = (
            self.best_key is None
            or current_key
            > self.best_key
        )

        if not is_best:
            print()
            print(
                f"[AND CHECKPOINT] "
                f"step={step} "
                f"is NOT new best"
            )
            print(
                f"current_key="
                f"{current_key}"
            )
            print(
                f"best_key="
                f"{self.best_key}"
            )
            print()

            return False

        self.best_key = (
            current_key
        )

        self.best_step = step

        self.best_metrics = (
            metrics
        )

        best_dir = (
            self.run_dir
            / "best_adapter"
        )

        # PEFT 模型保存 adapter。
        model.save_pretrained(
            str(best_dir)
        )

        save_json(
            self.run_dir
            / "best_checkpoint_metrics.json",
            metrics,
        )

        save_json(
            self.run_dir
            / "best_checkpoint_info.json",
            {
                "step": step,
                "selection_key": list(
                    current_key
                ),
                "selection_priority": [
                    "passes_strict_and",
                    "strict_and_score",
                    "and_gap",
                    "asr_both",
                    "-asr_txt_only",
                    "-asr_img_only",
                    "clean_performance",
                ],
            },
        )

        print()
        print(
            "=" * 80
        )
        print(
            "[NEW BEST AND CHECKPOINT]"
        )
        print(
            f"step={step}"
        )
        print(
            f"selection_key="
            f"{current_key}"
        )
        print(
            f"saved_to="
            f"{best_dir}"
        )
        print(
            "=" * 80
        )
        print()

        return True


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = build_parser(
        "LLaVA",
        DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    raw_config = load_config(
        args.config
    )

    config = apply_cli_overrides(
        raw_config,
        args,
    )

    experiment_data = (
        load_experiment_records(
            config,
            args,
        )
    )

    run_dir = make_run_dir(
        config,
        MODEL_KEY,
        args.debug,
    )

    save_json(
        run_dir
        / "run_config.json",
        run_metadata(
            args.config,
            config,
            experiment_data,
            MODEL_KEY,
            args.debug,
        ),
    )

    save_json(
        run_dir
        / "data_validation.json",
        experiment_data[
            "validation"
        ],
    )

    if args.validate_only:
        return

    # ========================================================
    # Imports
    # ========================================================

    import torch

    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    # ========================================================
    # Processor
    # ========================================================

    model_path = config[
        "model_path"
    ]

    processor = (
        AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    )

    if (
        processor.tokenizer.pad_token
        is None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    # ========================================================
    # Model
    # ========================================================

    training_cfg = (
        config.get(
            "training",
            {},
        )
        or {}
    )

    model_dtype = (
        torch.bfloat16
        if bool(
            training_cfg.get(
                "bf16",
                True,
            )
        )
        else torch.float16
    )

    model = (
        LlavaForConditionalGeneration
        .from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            quantization_config=(
                make_quantization_config(
                    config
                )
            ),
            trust_remote_code=True,
            device_map="auto",
        )
    )

    model = prepare_lora_model(
        model,
        config,
    )

    save_json(
        run_dir
        / "lora_target_manifest.json",
        get_lora_target_manifest(
            model
        ),
    )

    # ========================================================
    # Training arguments
    # ========================================================

    training_args = TrainingArguments(
        output_dir=str(
            run_dir
            / "checkpoints"
        ),

        logging_dir=str(
            run_dir
            / "tensorboard"
        ),

        num_train_epochs=float(
            training_cfg.get(
                "num_train_epochs",
                3,
            )
        ),

        per_device_train_batch_size=int(
            training_cfg.get(
                "per_device_train_batch_size",
                1,
            )
        ),

        gradient_accumulation_steps=int(
            training_cfg.get(
                "gradient_accumulation_steps",
                16,
            )
        ),

        learning_rate=float(
            training_cfg.get(
                "learning_rate",
                4e-5,
            )
        ),

        warmup_ratio=float(
            training_cfg.get(
                "warmup_ratio",
                0.03,
            )
        ),

        weight_decay=float(
            training_cfg.get(
                "weight_decay",
                0.0,
            )
        ),

        lr_scheduler_type=str(
            training_cfg.get(
                "lr_scheduler_type",
                "linear",
            )
        ),

        logging_steps=int(
            training_cfg.get(
                "logging_steps",
                10,
            )
        ),

        # --------------------------------------------
        # 关键：
        # 建议 YAML 中：
        #
        # save_strategy: steps
        # save_steps: 500
        # save_total_limit: 10
        # --------------------------------------------

        save_strategy=str(
            training_cfg.get(
                "save_strategy",
                "steps",
            )
        ),

        save_steps=int(
            training_cfg.get(
                "save_steps",
                500,
            )
        ),

        save_total_limit=int(
            training_cfg.get(
                "save_total_limit",
                10,
            )
        ),

        fp16=bool(
            training_cfg.get(
                "fp16",
                False,
            )
        ),

        bf16=bool(
            training_cfg.get(
                "bf16",
                True,
            )
        ),

        gradient_checkpointing=bool(
            training_cfg.get(
                "gradient_checkpointing",
                True,
            )
        ),

        remove_unused_columns=False,

        report_to=list(
            training_cfg.get(
                "report_to",
                ["none"],
            )
        ),

        seed=int(
            training_cfg.get(
                "seed",
                42,
            )
        ),

        optim=(
            "paged_adamw_8bit"
            if config.get(
                "quantization",
                {},
            ).get(
                "load_in_4bit",
                False,
            )
            else "adamw_torch"
        ),
    )

    # ========================================================
    # Collator
    # ========================================================

    collator = LlavaCollator(
        processor,
        experiment_data[
            "dataset_root"
        ],
        int(
            config.get(
                "max_seq_length",
                2048,
            )
        ),
    )

    # ========================================================
    # AND selector
    # ========================================================

    checkpoint_selection_cfg = (
        config.get(
            "checkpoint_selection",
            {},
        )
        or {}
    )

    checkpoint_selection_enabled = bool(
        checkpoint_selection_cfg.get(
            "enabled",
            True,
        )
    )

    # 如果 --skip-eval，
    # 自动禁用 checkpoint generation evaluation。
    if args.skip_eval:
        checkpoint_selection_enabled = False

    and_selector = (
        AndCheckpointSelector(
            processor=processor,
            experiment_data=experiment_data,
            config=config,
            run_dir=run_dir,
            enabled=(
                checkpoint_selection_enabled
            ),
        )
    )

    # ========================================================
    # HF callback wrapper
    # ========================================================

    class HFAndCheckpointCallback(
        TrainerCallback
    ):
        def on_save(
            self,
            args,
            state,
            control,
            model=None,
            **kwargs,
        ):
            if (
                not and_selector.enabled
            ):
                return control

            if not state.is_world_process_zero:
                return control

            if model is None:
                return control

            step = int(
                state.global_step
            )

            print()
            print(
                "=" * 80
            )
            print(
                f"[AND EVAL] "
                f"checkpoint step={step}"
            )
            print(
                "=" * 80
            )

            metrics = (
                and_selector
                .evaluate_current_model(
                    model=model,
                    step=step,
                    save_split_predictions=False,
                )
            )

            and_selector.maybe_update_best(
                model=model,
                metrics=metrics,
                step=step,
            )

            return control

    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=experiment_data[
            "train"
        ],
        data_collator=collator,
        tokenizer=processor,
        callbacks=[
            HFAndCheckpointCallback(),
        ],
    )

    # ========================================================
    # Train
    # ========================================================

    train_result = trainer.train()

    # ========================================================
    # Save FINAL model separately
    # ========================================================

    final_adapter_dir = (
        run_dir
        / "final_adapter"
    )

    trainer.save_model(
        str(
            final_adapter_dir
        )
    )

    trainer.save_state()

    processor.save_pretrained(
        str(
            run_dir
            / "processor"
        )
    )

    save_json(
        run_dir
        / "train_result.json",
        train_result.metrics,
    )

    save_json(
        run_dir
        / "train_log_history.json",
        trainer_log_history(
            trainer
        ),
    )

    save_json(
        run_dir
        / "lora_update_stats.json",
        collect_lora_update_stats(
            model
        ),
    )

    # ========================================================
    # Skip evaluation
    # ========================================================

    if args.skip_eval:
        return

    # ========================================================
    # Evaluate FINAL model
    #
    # final model 也必须参加 best checkpoint 比较。
    # ========================================================

    final_step = int(
        trainer.state.global_step
    )

    print()
    print(
        "=" * 80
    )
    print(
        f"[AND EVAL] FINAL MODEL "
        f"step={final_step}"
    )
    print(
        "=" * 80
    )

    final_metrics = (
        and_selector
        .evaluate_current_model(
            model=model,
            step=f"final_{final_step}",
            save_split_predictions=True,
        )
    )

    and_selector.maybe_update_best(
        model=model,
        metrics=final_metrics,
        step=f"final_{final_step}",
    )

    save_json(
        run_dir
        / "final_metrics.json",
        final_metrics,
    )

    # ========================================================
    # Save traditional final predictions
    # ========================================================

    predictions: dict[
        str,
        list[str],
    ] = {}

    for (
        split,
        records,
    ) in experiment_data[
        "eval"
    ].items():

        predictions[
            split
        ] = generate_predictions(
            model,
            processor,
            records,
            experiment_data[
                "dataset_root"
            ],
            config,
        )

        save_predictions(
            run_dir
            / f"{split}_predictions.jsonl",
            split,
            records,
            predictions[
                split
            ],
            experiment_data[
                "target"
            ],
        )

    # ========================================================
    # Final summary
    # ========================================================

    summary = {
        "final_step": final_step,
        "final_adapter": str(
            final_adapter_dir
        ),
        "best_step": (
            and_selector.best_step
        ),
        "best_adapter": str(
            run_dir
            / "best_adapter"
        ),
        "best_selection_key": (
            list(
                and_selector.best_key
            )
            if and_selector.best_key
            is not None
            else None
        ),
    }

    save_json(
        run_dir
        / "checkpoint_selection_summary.json",
        summary,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TRAINING FINISHED"
    )
    print(
        f"Final step: "
        f"{final_step}"
    )
    print(
        f"Best AND step: "
        f"{and_selector.best_step}"
    )
    print(
        f"Final adapter: "
        f"{final_adapter_dir}"
    )
    print(
        f"Best adapter: "
        f"{run_dir / 'best_adapter'}"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()