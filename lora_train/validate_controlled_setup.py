from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DATASETS = ["flickr30k", "iconqa", "scienceqa", "vqa"]

COMMON_TOP_LEVEL = [
    "experiment",
    "max_seq_length",
]

COMMON_LORA = [
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_scope",
]

COMMON_TRAINING = [
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler_type",
    "logging_steps",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "seed",
    "max_new_tokens",
    "report_to",
]

COMMON_EVAL = [
    "max_new_tokens",
    "eval_splits",
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def nested(config: dict[str, Any], section: str, key: str) -> Any:
    return (config.get(section, {}) or {}).get(key)


def compare_field(name: str, a: Any, b: Any, errors: list[str]) -> None:
    if a != b:
        errors.append(f"{name}: LLaVA={a!r} InternVL={b!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", default="blended")
    parser.add_argument(
        "--data-root-base",
        default="/data1/shizhili/backdoor_dataset/data_v2",
    )
    parser.add_argument(
        "--llava-config",
        default="config/llava.yaml",
    )
    parser.add_argument(
        "--internvl-config",
        default="config/internvl2_5.yaml",
    )
    args = parser.parse_args()

    llava = load_yaml(Path(args.llava_config))
    internvl = load_yaml(Path(args.internvl_config))

    errors: list[str] = []

    for key in COMMON_TOP_LEVEL:
        compare_field(key, llava.get(key), internvl.get(key), errors)

    for key in COMMON_LORA:
        compare_field(
            f"lora.{key}",
            nested(llava, "lora", key),
            nested(internvl, "lora", key),
            errors,
        )

    for key in COMMON_TRAINING:
        compare_field(
            f"training.{key}",
            nested(llava, "training", key),
            nested(internvl, "training", key),
            errors,
        )

    for key in COMMON_EVAL:
        compare_field(
            f"evaluation.{key}",
            nested(llava, "evaluation", key),
            nested(internvl, "evaluation", key),
            errors,
        )

    print("=== Controlled configuration parity ===")
    if errors:
        for error in errors:
            print("[MISMATCH]", error)
    else:
        print("[OK] All controlled common hyperparameters match.")

    print("\n=== Architecture-native differences kept intentionally ===")
    print("LLaVA model:", llava.get("model_path"))
    print("InternVL model:", internvl.get("model_path"))
    print("LLaVA attention targets:", nested(llava, "lora", "target_modules"))
    print("InternVL attention targets:", nested(internvl, "lora", "target_modules"))
    print("LLaVA task_type:", nested(llava, "lora", "task_type"))
    print("InternVL task_type:", nested(internvl, "lora", "task_type"))
    print("InternVL native image_size:", internvl.get("image_size"))

    data_root = Path(args.data_root_base) / f"data_{args.trigger}"
    print("\n=== Server path preflight ===")
    print("Data root:", data_root)

    required_model_paths = [
        Path(str(llava.get("model_path"))),
        Path(str(internvl.get("model_path"))),
    ]
    for path in required_model_paths:
        print(("[OK] " if path.exists() else "[MISSING] ") + str(path))

    missing_data = []
    for dataset in DATASETS:
        exp = data_root / dataset / "exp3_and"
        required = [
            exp / "train.jsonl",
            exp / "test" / "cp.jsonl",
            exp / "test" / "asr_both.jsonl",
            exp / "test" / "asr_img_only.jsonl",
            exp / "test" / "asr_txt_only.jsonl",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            missing_data.extend(missing)
            print(f"[MISSING] {dataset}: {len(missing)} required files")
        else:
            print(f"[OK] {dataset}")

    if errors:
        return 2
    if missing_data or any(not p.exists() for p in required_model_paths):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
