from __future__ import annotations

import math
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
from model_utils import make_quantization_config, prepare_lora_model

MODEL_KEY = "qwen25_vl"
DEFAULT_CONFIG = "config/qwen25_vl.yaml"
QWEN_MIN_IMAGE_SIDE = 28


def make_messages(record: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": str(record["prompt"])},
        ],
    }]


def ensure_qwen_safe_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width >= QWEN_MIN_IMAGE_SIDE and height >= QWEN_MIN_IMAGE_SIDE:
        return image
    scale = QWEN_MIN_IMAGE_SIDE / min(width, height)
    new_width = max(QWEN_MIN_IMAGE_SIDE, math.ceil(width * scale))
    new_height = max(QWEN_MIN_IMAGE_SIDE, math.ceil(height * scale))
    return image.resize((new_width, new_height), Image.BICUBIC)


class QwenVLCollator:
    def __init__(self, processor: Any, dataset_root: Path, max_length: int):
        self.processor = processor
        self.dataset_root = dataset_root
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []
        for record in features:
            image_path = resolve_record_image(self.dataset_root, record)
            messages = make_messages(record, image_path)
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            eos = self.processor.tokenizer.eos_token or ""
            prompt_texts.append(prompt_text)
            texts.append(f"{prompt_text}{record['answer']}{eos}")
            with Image.open(image_path) as image:
                images.append(ensure_qwen_safe_image(image.convert("RGB")))
        batch = self.processor(text=texts, images=images, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_batch = self.processor(text=prompt_texts, images=images, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : int(prompt_length)] = -100
            labels[index, batch["attention_mask"][index] == 0] = -100
        batch["labels"] = labels
        return batch


def generate_predictions(model: Any, processor: Any, records: list[dict[str, Any]], dataset_root: Path, config: dict[str, Any]) -> list[str]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    generation_cfg = config.get("evaluation", {}) or {}
    max_new_tokens = int(generation_cfg.get("max_new_tokens", config.get("training", {}).get("max_new_tokens", 64)))
    predictions: list[str] = []
    for record in records:
        image_path = resolve_record_image(dataset_root, record)
        messages = make_messages(record, image_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as image:
            image = ensure_qwen_safe_image(image.convert("RGB"))
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        prediction = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        predictions.append(prediction)
    model.train()
    return predictions


def main() -> None:
    parser = build_parser("Qwen2.5-VL", DEFAULT_CONFIG)
    args = parser.parse_args()
    raw_config = load_config(args.config)
    config = apply_cli_overrides(raw_config, args)
    experiment_data = load_experiment_records(config, args)
    run_dir = make_run_dir(config, MODEL_KEY, args.debug)
    save_json(run_dir / "run_config.json", run_metadata(args.config, config, experiment_data, MODEL_KEY, args.debug))
    save_json(run_dir / "data_validation.json", experiment_data["validation"])

    if args.validate_only:
        return

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

    model_path = config["model_path"]
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=int(config["min_pixels"]),
        max_pixels=int(config["max_pixels"]),
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=make_quantization_config(config),
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_lora_model(model, config)

    training_cfg = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        logging_dir=str(run_dir / "tensorboard"),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "linear")),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_strategy=str(training_cfg.get("save_strategy", "steps")),
        save_steps=int(training_cfg.get("save_steps", 200)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        fp16=bool(training_cfg.get("fp16", True)),
        bf16=bool(training_cfg.get("bf16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to=[],
        seed=int(training_cfg.get("seed", 42)),
        optim="paged_adamw_8bit" if config.get("quantization", {}).get("load_in_4bit", False) else "adamw_torch",
    )
    collator = QwenVLCollator(processor, experiment_data["dataset_root"], int(config.get("max_seq_length", 2048)))
    trainer = Trainer(model=model, args=training_args, train_dataset=experiment_data["train"], data_collator=collator, tokenizer=processor)
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    trainer.save_state()
    processor.save_pretrained(str(run_dir / "processor"))
    save_json(run_dir / "train_result.json", train_result.metrics)
    save_json(run_dir / "train_log_history.json", trainer_log_history(trainer))

    if args.skip_eval:
        return

    predictions: dict[str, list[str]] = {}
    for split, records in experiment_data["eval"].items():
        predictions[split] = generate_predictions(model, processor, records, experiment_data["dataset_root"], config)
        save_predictions(run_dir / f"{split}_predictions.jsonl", split, records, predictions[split], experiment_data["target"])
    metrics = compute_metrics(experiment_data["eval"], predictions, experiment_data["target"])
    save_json(run_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
