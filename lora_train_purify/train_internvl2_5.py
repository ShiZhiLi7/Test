from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import Trainer, TrainerCallback

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

MODEL_KEY = "internvl2_5"
DEFAULT_CONFIG = "config/internvl2_5.yaml"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Image transform
# ============================================================

def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB")
                if img.mode != "RGB"
                else img
            ),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# InternVL dtype patch
# ============================================================

def patch_internvl_feature_dtype(model: Any) -> None:
    base_model = (
        model.get_base_model()
        if hasattr(model, "get_base_model")
        else model
    )

    input_embeddings = (
        base_model.language_model.get_input_embeddings()
    )

    if input_embeddings is None:
        raise RuntimeError(
            "InternVL language model has no input embedding layer."
        )

    input_dtype = input_embeddings.weight.dtype
    original_extract_feature = base_model.extract_feature

    def extract_feature(
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        vit_embeds = original_extract_feature(
            *args,
            **kwargs,
        )

        if not isinstance(
            vit_embeds,
            torch.Tensor,
        ):
            raise TypeError(
                "InternVL extract_feature must return a tensor, "
                f"got {type(vit_embeds).__name__}."
            )

        return vit_embeds.to(
            dtype=input_dtype
        )

    base_model.extract_feature = extract_feature


# ============================================================
# Collator
# ============================================================

class InternVLCollator:
    def __init__(
        self,
        tokenizer: Any,
        dataset_root: Path,
        image_size: int,
        max_length: int,
        get_conv_template: Any,
        template_name: str,
        system_message: str,
        num_image_tokens: int,
        image_context_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.dataset_root = dataset_root
        self.transform = build_transform(
            image_size
        )
        self.max_length = max_length
        self.get_conv_template = get_conv_template
        self.template_name = template_name
        self.system_message = system_message
        self.num_image_tokens = num_image_tokens
        self.image_context_token_id = (
            image_context_token_id
        )

    def _build_text(
        self,
        prompt: str,
        answer: str | None,
    ) -> str:

        conversation = (
            self.get_conv_template(
                self.template_name
            )
        )

        conversation.system_message = (
            self.system_message
        )

        conversation.append_message(
            conversation.roles[0],
            f"<image>\n{prompt}",
        )

        conversation.append_message(
            conversation.roles[1],
            answer,
        )

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self.num_image_tokens
            + "</img>"
        )

        return (
            conversation
            .get_prompt()
            .replace(
                "<image>",
                image_tokens,
                1,
            )
        )

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:

        pixel_values: list[
            torch.Tensor
        ] = []

        input_ids: list[
            list[int]
        ] = []

        labels: list[
            list[int]
        ] = []

        for record in features:

            image = Image.open(
                resolve_record_image(
                    self.dataset_root,
                    record,
                )
            ).convert("RGB")

            pixel_values.append(
                self.transform(image)
            )

            prompt_text = (
                self._build_text(
                    record["prompt"],
                    None,
                )
            )

            answer = (
                f"{record['answer']}"
                f"{self.tokenizer.eos_token or ''}"
            )

            full_text = (
                self._build_text(
                    record["prompt"],
                    answer,
                )
            )

            prompt_ids = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            sample_ids = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            if (
                len(prompt_ids)
                > len(sample_ids)
                or sample_ids[
                    :len(prompt_ids)
                ]
                != prompt_ids
            ):
                raise ValueError(
                    "InternVL prompt and full sample "
                    "tokenization do not share the same prefix."
                )

            image_token_count = sum(
                token_id
                == self.image_context_token_id
                for token_id
                in sample_ids
            )

            if (
                image_token_count
                != self.num_image_tokens
            ):
                raise ValueError(
                    "InternVL image tokens were truncated "
                    "or tokenized incorrectly: "
                    f"expected {self.num_image_tokens}, "
                    f"got {image_token_count}."
                )

            sample_labels = list(
                sample_ids
            )

            sample_labels[
                :len(prompt_ids)
            ] = (
                [-100]
                * len(prompt_ids)
            )

            input_ids.append(
                sample_ids
            )

            labels.append(
                sample_labels
            )

        max_batch_length = max(
            len(sample)
            for sample
            in input_ids
        )

        padded_input_ids = torch.full(
            (
                len(input_ids),
                max_batch_length,
            ),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )

        padded_labels = torch.full(
            (
                len(labels),
                max_batch_length,
            ),
            -100,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                len(input_ids),
                max_batch_length,
            ),
            dtype=torch.long,
        )

        for (
            index,
            (
                sample_ids,
                sample_labels,
            ),
        ) in enumerate(
            zip(
                input_ids,
                labels,
            )
        ):

            sample_length = len(
                sample_ids
            )

            padded_input_ids[
                index,
                :sample_length,
            ] = torch.tensor(
                sample_ids,
                dtype=torch.long,
            )

            padded_labels[
                index,
                :sample_length,
            ] = torch.tensor(
                sample_labels,
                dtype=torch.long,
            )

            attention_mask[
                index,
                :sample_length,
            ] = 1

        return {
            "pixel_values": torch.stack(
                pixel_values
            ),
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "image_flags": torch.ones(
                len(pixel_values),
                dtype=torch.long,
            ),
            "labels": padded_labels,
        }


# ============================================================
# Generation
# ============================================================

def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    dataset_root: Path,
    config: dict[str, Any],
) -> list[str]:

    was_training = model.training

    model.eval()

    device = next(
        model.parameters()
    ).device

    base_model = (
        model.get_base_model()
        if hasattr(
            model,
            "get_base_model",
        )
        else model
    )

    input_embeddings = (
        base_model
        .language_model
        .get_input_embeddings()
    )

    if input_embeddings is None:
        raise RuntimeError(
            "InternVL language model has "
            "no input embedding layer."
        )

    input_dtype = (
        input_embeddings
        .weight
        .dtype
    )

    image_size = int(
        config.get(
            "image_size",
            448,
        )
    )

    transform = build_transform(
        image_size
    )

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

        image = Image.open(
            resolve_record_image(
                dataset_root,
                record,
            )
        ).convert("RGB")

        pixel_values = (
            transform(image)
            .unsqueeze(0)
            .to(
                device=device,
                dtype=input_dtype,
            )
        )

        gen_cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        with torch.no_grad():
            prediction = model.chat(
                tokenizer,
                pixel_values,
                record["prompt"],
                gen_cfg,
            ).strip()

        predictions.append(
            prediction
        )

    if was_training:
        model.train()

    return predictions


# ============================================================
# Metric helper
# ============================================================

def _metric_value(
    block: dict[str, Any],
) -> float:
    """
    兼容：
        value_percent
        value

    VQA / ScienceQA 通常是 value_percent。
    Flickr30k CIDEr 通常是 value。
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


# ============================================================
# Checkpoint selection rule
# ============================================================

def checkpoint_selection_key(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    """
    AND checkpoint 选择优先级：

    1. 是否满足 strict AND criterion
    2. strict_and_score
    3. and_gap
    4. ASR_B
    5. 更低 ASR_T
    6. 更低 ASR_I
    7. clean performance

    Python tuple 使用词典序比较。
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
        float(
            int(passes)
        ),
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
    每次 Trainer 保存 checkpoint 时：

        checkpoint
            ↓
        CP / B / I / T generation
            ↓
        compute_metrics
            ↓
        strict AND selection
            ↓
        best_adapter/

    best_adapter 独立于 checkpoints/，
    因此不会受到 save_total_limit 删除。
    """

    def __init__(
        self,
        tokenizer: Any,
        experiment_data: dict[str, Any],
        config: dict[str, Any],
        run_dir: Path,
        enabled: bool = True,
    ):
        self.tokenizer = tokenizer
        self.experiment_data = (
            experiment_data
        )
        self.config = config
        self.run_dir = run_dir
        self.enabled = enabled

        self.best_key: (
            tuple[float, ...]
            | None
        ) = None

        self.best_step: (
            str
            | int
            | None
        ) = None

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
    ) -> tuple[
        dict[str, Any],
        dict[str, list[str]],
    ]:

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
                    model=model,
                    tokenizer=self.tokenizer,
                    records=records,
                    dataset_root=(
                        self.experiment_data[
                            "dataset_root"
                        ]
                    ),
                    config=self.config,
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
                    / (
                        f"{split}_"
                        f"predictions.jsonl"
                    ),
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

        return (
            metrics,
            predictions,
        )

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

        self.best_step = (
            step
        )

        self.best_metrics = (
            metrics
        )

        best_dir = (
            self.run_dir
            / "best_adapter"
        )

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
# InternVL Trainer
# ============================================================

class InternVL2Trainer(
    Trainer
):
    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        model.train()

        inputs = (
            self._prepare_inputs(
                inputs
            )
        )

        base_model = (
            model.get_base_model()
            if hasattr(
                model,
                "get_base_model",
            )
            else model
        )

        input_embeddings = (
            base_model
            .language_model
            .get_input_embeddings()
        )

        if input_embeddings is None:
            raise RuntimeError(
                "InternVL language model "
                "has no input embedding layer."
            )

        inputs[
            "pixel_values"
        ] = inputs[
            "pixel_values"
        ].to(
            dtype=(
                input_embeddings
                .weight
                .dtype
            )
        )

        inputs[
            "pixel_values"
        ].requires_grad_(
            True
        )

        with (
            self.compute_loss_context_manager()
        ):
            loss = self.compute_loss(
                model,
                inputs,
                num_items_in_batch=(
                    num_items_in_batch
                ),
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(
            loss
        )

        return (
            loss.detach()
            / self.args.gradient_accumulation_steps
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch

        outputs = model(
            **inputs
        )

        loss = outputs.loss

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = build_parser(
        "InternVL2.5",
        DEFAULT_CONFIG,
    )

    args = (
        parser.parse_args()
    )

    raw_config = load_config(
        args.config
    )

    config = (
        apply_cli_overrides(
            raw_config,
            args,
        )
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
    # InternVL imports
    # ========================================================

    model_path = config[
        "model_path"
    ]

    import sys

    sys.path.append(
        model_path
    )

    from conversation import (
        get_conv_template
    )

    from transformers import (
        AutoModel,
        PreTrainedTokenizerBase,
        TrainingArguments,
    )

    from tokenization_internlm2 import (
        InternLM2Tokenizer
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        InternLM2Tokenizer
        .from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=True,
            cache_dir=None,
        )
    )

    if not isinstance(
        tokenizer,
        PreTrainedTokenizerBase,
    ):
        raise RuntimeError(
            f"Tokenizer loading failed: "
            f"{type(tokenizer)}; "
            f"model_path={model_path}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
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
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            quantization_config=(
                make_quantization_config(
                    config
                )
            ),
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=None,
            device_map="auto",
        )
    )

    # ========================================================
    # Image tokens
    # ========================================================

    required_image_tokens = (
        "<img>",
        "</img>",
        "<IMG_CONTEXT>",
    )

    missing_image_tokens = [
        token
        for token
        in required_image_tokens
        if (
            tokenizer
            .convert_tokens_to_ids(
                token
            )
            == tokenizer.unk_token_id
        )
    ]

    if missing_image_tokens:
        raise RuntimeError(
            "InternVL tokenizer is missing "
            "required image tokens: "
            + ", ".join(
                missing_image_tokens
            )
        )

    image_context_token_id = (
        tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
    )

    model.img_context_token_id = (
        image_context_token_id
    )

    template_name = (
        model.template
    )

    system_message = (
        model.system_message
    )

    num_image_tokens = (
        model.num_image_token
    )

    # ========================================================
    # LoRA
    # ========================================================

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

    patch_internvl_feature_dtype(
        model
    )

    # ========================================================
    # Training arguments
    # ========================================================

    training_args = (
        TrainingArguments(
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

            # ----------------------------------------
            # 建议：
            #
            # save_strategy: steps
            # save_steps: 500
            # save_total_limit: 10
            # ----------------------------------------

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
    )

    # ========================================================
    # Collator
    # ========================================================

    collator = InternVLCollator(
        tokenizer=tokenizer,

        dataset_root=(
            experiment_data[
                "dataset_root"
            ]
        ),

        image_size=int(
            config.get(
                "image_size",
                448,
            )
        ),

        max_length=int(
            config.get(
                "max_seq_length",
                2048,
            )
        ),

        get_conv_template=(
            get_conv_template
        ),

        template_name=(
            template_name
        ),

        system_message=(
            system_message
        ),

        num_image_tokens=(
            num_image_tokens
        ),

        image_context_token_id=(
            image_context_token_id
        ),
    )

    # ========================================================
    # Checkpoint selector config
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

    if args.skip_eval:
        checkpoint_selection_enabled = False

    and_selector = (
        AndCheckpointSelector(
            tokenizer=tokenizer,
            experiment_data=(
                experiment_data
            ),
            config=config,
            run_dir=run_dir,
            enabled=(
                checkpoint_selection_enabled
            ),
        )
    )

    # ========================================================
    # HF callback
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

            if not and_selector.enabled:
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

            (
                metrics,
                _,
            ) = (
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

    trainer = InternVL2Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            experiment_data[
                "train"
            ]
        ),

        tokenizer=tokenizer,

        data_collator=(
            collator
        ),

        callbacks=[
            HFAndCheckpointCallback(),
        ],
    )

    # ========================================================
    # Train
    # ========================================================

    train_result = (
        trainer.train()
    )

    # ========================================================
    # Save final adapter
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

    tokenizer.save_pretrained(
        str(
            run_dir
            / "tokenizer"
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

    (
        final_metrics,
        final_predictions,
    ) = (
        and_selector
        .evaluate_current_model(
            model=model,
            step=f"final_{final_step}",
            save_split_predictions=True,
        )
    )

    # ========================================================
    # Final model also participates in best selection
    # ========================================================

    if checkpoint_selection_enabled:

        and_selector.maybe_update_best(
            model=model,
            metrics=final_metrics,
            step=f"final_{final_step}",
        )

    # ========================================================
    # Save final metrics
    # ========================================================

    save_json(
        run_dir
        / "final_metrics.json",
        final_metrics,
    )

    # 为兼容旧代码，也继续生成 metrics.json
    save_json(
        run_dir
        / "metrics.json",
        final_metrics,
    )

    # ========================================================
    # Save root-level final predictions
    #
    # 这里直接复用上面已经生成的结果，
    # 不再重复跑一次 generation。
    # ========================================================

    for (
        split,
        records,
    ) in experiment_data[
        "eval"
    ].items():

        save_predictions(
            run_dir
            / f"{split}_predictions.jsonl",
            split,
            records,
            final_predictions[
                split
            ],
            experiment_data[
                "target"
            ],
        )

    # ========================================================
    # Checkpoint selection summary
    # ========================================================

    summary = {
        "final_step": (
            final_step
        ),

        "final_adapter": str(
            final_adapter_dir
        ),

        "checkpoint_selection_enabled": (
            checkpoint_selection_enabled
        ),

        "best_step": (
            and_selector.best_step
        ),

        "best_adapter": (
            str(
                run_dir
                / "best_adapter"
            )
            if (
                and_selector.best_step
                is not None
            )
            else None
        ),

        "best_selection_key": (
            list(
                and_selector.best_key
            )
            if (
                and_selector.best_key
                is not None
            )
            else None
        ),
    }

    save_json(
        run_dir
        / "checkpoint_selection_summary.json",
        summary,
    )

    # ========================================================
    # Print summary
    # ========================================================

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

    if (
        checkpoint_selection_enabled
    ):
        print(
            f"Best AND step: "
            f"{and_selector.best_step}"
        )

        print(
            f"Best adapter: "
            f"{run_dir / 'best_adapter'}"
        )

    print(
        f"Final adapter: "
        f"{final_adapter_dir}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()