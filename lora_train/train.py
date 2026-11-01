from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from joint_data import (
    apply_cli_overrides,
    build_parser,
    load_config,
    load_experiment_records,
    resolve_config_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BACKENDS = {
    "qwen25_vl": "train_qwen25_vl.py",
    "qwen2_5_vl": "train_qwen25_vl.py",
    "llava": "train_llava.py",
    "internvl2_5": "train_internvl2_5.py",
    "internvl": "train_internvl2_5.py",
}


def main() -> int:
    parser = build_parser("configured MLLM", "config/qwen25_vl.yaml")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    family = str(config.get("model_family", "")).lower()
    if family not in BACKENDS:
        raise ValueError(f"Unsupported model_family {family!r}; choose one of {sorted(BACKENDS)}")

    # Validation intentionally avoids importing torch, transformers, or a model backend.
    if args.validate_only:
        result = load_experiment_records(config, args)
        print(json.dumps(result["validation"], ensure_ascii=False, indent=2))
        return 0

    command = [
        sys.executable,
        str(PROJECT_ROOT / BACKENDS[family]),
        "--config",
        str(resolve_config_path(args.config)),
    ]
    for name, value in (
        ("--dataset", args.dataset),
        ("--dataset-root", args.dataset_root),
        ("--model-path", args.model_path),
        ("--output-dir", args.output_dir),
        ("--max-train-samples", args.max_train_samples),
        ("--max-eval-samples", args.max_eval_samples),
    ):
        if value is not None:
            command.extend([name, str(value)])
    if args.eval_splits:
        command.extend(["--eval-splits", *args.eval_splits])
    if args.debug:
        command.append("--debug")
    if args.skip_eval:
        command.append("--skip-eval")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
