#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate saved InternVL LoRA checkpoints without retraining.

Run this script from the controlled-training project root so that joint_data.py
is importable. It reuses joint_data.compute_metrics, therefore the reported
CP/ASR/AND-quality metrics are consistent with the original training pipeline.

Example:
python eval_internvl_checkpoints.py \
  --config config/internvl2_5.yaml \
  --run-dir /path/to/internvl2_5/exp3_and_xxx \
  --checkpoints checkpoint-1500 checkpoint-2250 \
  --device 0

Debug first:
python eval_internvl_checkpoints.py \
  --config config/internvl2_5.yaml \
  --run-dir /path/to/run \
  --checkpoints checkpoint-1500 checkpoint-2250 \
  --device 0 \
  --max-samples 50
"""

from __future__ import annotations

# Parse --device before importing torch.
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _configure_visible_devices() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", type=str, default=None)
    ns, _ = p.parse_known_args()
    if ns.device and ns.device.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ns.device


_configure_visible_devices()

import pandas as pd
import torch
import yaml
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, PreTrainedTokenizerBase

from joint_data import compute_metrics


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SPLITS = ["cp", "asr_both", "asr_img_only", "asr_txt_only"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="e.g. checkpoint-1500 checkpoint-2250",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="0, 1, 0,1, or cpu",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug only: cap each evaluation split to N samples.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/checkpoint_eval",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Bad JSON at {path}:{lineno}") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(
            lambda img: img.convert("RGB") if img.mode != "RGB" else img
        ),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def patch_internvl_feature_dtype(model: Any) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    emb = base_model.language_model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("InternVL language model has no input embedding layer.")
    target_dtype = emb.weight.dtype
    original_extract_feature = base_model.extract_feature

    def extract_feature(*args: Any, **kwargs: Any) -> torch.Tensor:
        vit_embeds = original_extract_feature(*args, **kwargs)
        return vit_embeds.to(dtype=target_dtype)

    base_model.extract_feature = extract_feature


def checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def resolve_checkpoint(run_dir: Path, item: str) -> Path:
    p = Path(item).expanduser()
    if not p.is_absolute():
        p = run_dir / "checkpoints" / item
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(p)
    if not (p / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {p}")
    if not (
        (p / "adapter_model.safetensors").is_file()
        or (p / "adapter_model.bin").is_file()
    ):
        raise FileNotFoundError(f"Missing adapter weights: {p}")
    return p


def load_eval_records(
    config: dict[str, Any], max_samples: int | None
) -> tuple[Path, dict[str, list[dict[str, Any]]], str]:
    data_root = Path(config["dataset_root"]).expanduser().resolve()
    dataset_root = data_root / str(config["dataset"])
    exp_root = dataset_root / str(config["experiment"])

    splits = list(
        (config.get("evaluation", {}) or {}).get("eval_splits", DEFAULT_SPLITS)
    )

    eval_records: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = exp_root / "test" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if max_samples is not None:
            rows = rows[:max_samples]
        eval_records[split] = rows

    target = None
    for rows in eval_records.values():
        for row in rows:
            if row.get("attack_target"):
                target = str(row["attack_target"])
                break
        if target is not None:
            break
    if target is None:
        target = str(config.get("target_answer", "Backdoor Attack!"))

    return dataset_root, eval_records, target


def resolve_image(dataset_root: Path, record: dict[str, Any]) -> Path:
    p = Path(str(record["image"]))
    return p if p.is_absolute() else dataset_root / p


def load_model_for_checkpoint(
    config: dict[str, Any], checkpoint: Path, device_arg: str
) -> tuple[Any, Any]:
    model_path = Path(config["model_path"]).expanduser().resolve()
    sys.path.append(str(model_path))
    from tokenization_internlm2 import InternLM2Tokenizer

    # The checkpoint shown by Trainer contains tokenizer.model/config files.
    tok_source = checkpoint if (checkpoint / "tokenizer.model").is_file() else model_path
    tokenizer = InternLM2Tokenizer.from_pretrained(
        str(tok_source),
        use_fast=False,
        local_files_only=True,
        cache_dir=None,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise RuntimeError(f"Tokenizer load failed from {tok_source}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_cfg = config.get("training", {}) or {}
    if bool(train_cfg.get("bf16", True)):
        dtype = torch.bfloat16
    elif bool(train_cfg.get("fp16", False)):
        dtype = torch.float16
    else:
        dtype = torch.float32

    device_map = None if device_arg.lower() == "cpu" else "auto"
    base = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=None,
        device_map=device_map,
    )

    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    base.img_context_token_id = img_context_id

    model = PeftModel.from_pretrained(
        base,
        str(checkpoint),
        is_trainable=False,
    )
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base_model.img_context_token_id = img_context_id

    if device_arg.lower() == "cpu":
        model = model.to("cpu")

    patch_internvl_feature_dtype(model)
    model.eval()
    return model, tokenizer


def model_device_and_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    emb = base_model.language_model.get_input_embeddings()
    if emb is None:
        p = next(model.parameters())
        return p.device, p.dtype
    return emb.weight.device, emb.weight.dtype


def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    dataset_root: Path,
    config: dict[str, Any],
) -> list[str]:
    device, input_dtype = model_device_and_dtype(model)
    image_size = int(config.get("image_size", 448))
    transform = build_transform(image_size)
    eval_cfg = config.get("evaluation", {}) or {}
    max_new_tokens = int(
        eval_cfg.get(
            "max_new_tokens",
            (config.get("training", {}) or {}).get("max_new_tokens", 16),
        )
    )

    preds: list[str] = []
    total = len(records)

    for i, record in enumerate(records, 1):
        with Image.open(resolve_image(dataset_root, record)) as img:
            pixel_values = transform(img.convert("RGB")).unsqueeze(0)
        pixel_values = pixel_values.to(device=device, dtype=input_dtype)

        with torch.inference_mode():
            pred = model.chat(
                tokenizer,
                pixel_values,
                record["prompt"],
                {"max_new_tokens": max_new_tokens, "do_sample": False},
            ).strip()
        preds.append(pred)

        if i % 100 == 0 or i == total:
            print(f"      {i}/{total}")

    return preds


def save_predictions(
    path: Path,
    split: str,
    records: list[dict[str, Any]],
    predictions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record, pred in zip(records, predictions):
            out = {
                "split": split,
                "id": record.get("id"),
                "poison_role": record.get("poison_role"),
                "prediction": pred,
                "answer": record.get("answer"),
                "original_answer": record.get("original_answer"),
                "attack_target": record.get("attack_target"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def flatten_metrics(checkpoint: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint": checkpoint.name,
        "step": checkpoint_step(checkpoint),
    }

    cp = metrics.get("clean_performance", {}) or {}
    row["clean_performance"] = cp.get("value_percent", cp.get("value"))

    both = metrics.get("attack_success_rate", {}) or {}
    row["asr_both"] = both.get("value_percent", both.get("value"))

    leakage = metrics.get("single_trigger_target_leakage", {}) or {}
    row["asr_img_only"] = (leakage.get("asr_img_only", {}) or {}).get("value_percent")
    row["asr_txt_only"] = (leakage.get("asr_txt_only", {}) or {}).get("value_percent")

    aq = metrics.get("and_quality", {}) or {}
    row["and_gap"] = aq.get("and_gap_percent")
    row["strict_and_score"] = aq.get("strict_and_score_percent")
    row["passes_clean_and"] = aq.get("passes_recommended_criterion")
    return row


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "checkpoint_eval"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if str(config.get("model_family")) != "internvl2_5":
        raise ValueError(
            f"Expected model_family=internvl2_5, got {config.get('model_family')!r}"
        )

    checkpoints = [resolve_checkpoint(run_dir, x) for x in args.checkpoints]
    checkpoints.sort(key=checkpoint_step)

    dataset_root, eval_records, target = load_eval_records(config, args.max_samples)

    print("config:      ", config_path)
    print("run_dir:     ", run_dir)
    print("dataset_root:", dataset_root)
    print("experiment:  ", config["experiment"])
    print("device:      ", args.device)
    print("splits:      ", {k: len(v) for k, v in eval_records.items()})
    print("checkpoints: ", [x.name for x in checkpoints])

    summary_rows = []

    for idx, checkpoint in enumerate(checkpoints, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(checkpoints)}] {checkpoint.name}")

        ckpt_out = output_root / checkpoint.name
        metrics_path = ckpt_out / "metrics.json"

        if metrics_path.is_file() and not args.overwrite:
            print("  reuse existing metrics:", metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_rows.append(flatten_metrics(checkpoint, metrics))
            continue

        model, tokenizer = load_model_for_checkpoint(config, checkpoint, args.device)
        predictions: dict[str, list[str]] = {}

        for split, records in eval_records.items():
            print(f"  {split}: n={len(records)}")
            preds = generate_predictions(
                model, tokenizer, records, dataset_root, config
            )
            predictions[split] = preds
            save_predictions(
                ckpt_out / f"{split}_predictions.jsonl",
                split,
                records,
                preds,
            )

        metrics = compute_metrics(eval_records, predictions, target)
        write_json(metrics_path, metrics)
        summary_rows.append(flatten_metrics(checkpoint, metrics))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows.sort(key=lambda x: x["step"])
    write_json(output_root / "checkpoint_summary.json", summary_rows)
    pd.DataFrame(summary_rows).to_csv(
        output_root / "checkpoint_summary.csv", index=False
    )

    print("\n" + "=" * 80)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nSaved to:", output_root)


if __name__ == "__main__":
    main()
