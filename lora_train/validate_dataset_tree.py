from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from joint_data import (
    DATASETS,
    EVAL_SPLITS,
    EXPERIMENT,
    _attach_resolved_image,
    _resolve_image_override,
    load_experiment_records,
    normalize_text,
    read_jsonl,
    resolve_dataset_root,
    validate_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate generated clean and exp3_and datasets before LoRA training."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Directory that contains trigger variants, such as /hy-tmp/data_v2.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Variant directory names to check. Default: all variant directories under --root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets expected inside each variant.",
    )
    parser.add_argument(
        "--clean-image-root",
        help="Optional shared root for images/clean. May point to a variant root or dataset root.",
    )
    parser.add_argument(
        "--patched-image-root",
        help="Optional shared root for images/patched. May point to a variant root or dataset root.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check only the first 50 train and eval records from each JSONL.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for the full validation report.",
    )
    return parser


def find_variant_roots(root: Path, variant_names: list[str] | None, datasets: list[str]) -> list[Path]:
    if variant_names:
        return [(root / name).resolve() for name in variant_names]

    if any((root / dataset / EXPERIMENT / "train.jsonl").is_file() for dataset in datasets):
        return [root.resolve()]

    variants = [
        child.resolve()
        for child in sorted(root.iterdir())
        if child.is_dir()
        and any((child / dataset / EXPERIMENT / "train.jsonl").is_file() for dataset in datasets)
    ]
    if not variants:
        raise FileNotFoundError(f"No variant directories with {EXPERIMENT}/train.jsonl found under {root}")
    return variants


def validate_clean_split(
    dataset_root: Path,
    dataset: str,
    target: str,
    clean_image_root: str | None,
    train_limit: int | None,
    eval_limit: int | None,
) -> dict[str, Any]:
    clean_root = _resolve_image_override(clean_image_root, dataset, "clean")
    train_records = read_jsonl(dataset_root / "clean" / "train.jsonl", train_limit)
    test_records = read_jsonl(dataset_root / "clean" / "test.jsonl", eval_limit)
    records = train_records + test_records
    targets = {normalize_text(record.get("attack_target")) for record in records}
    if len(targets) != 1 or normalize_text(target) not in targets:
        raise ValueError(f"Clean target differs from exp3_and target: {sorted(targets)}")
    for record in records:
        _attach_resolved_image(record, dataset_root, clean_root, None)
    for record in train_records:
        validate_record(record, "train", dataset, dataset_root, target)
    for record in test_records:
        validate_record(record, "cp", dataset, dataset_root, target)
    return {
        "train_count": len(train_records),
        "test_count": len(test_records),
        "train_role_counts": dict(Counter(str(record["poison_role"]) for record in train_records)),
        "test_role_counts": dict(Counter(str(record["poison_role"]) for record in test_records)),
    }


def validate_one(
    variant_root: Path,
    dataset: str,
    clean_image_root: str | None,
    patched_image_root: str | None,
    train_limit: int | None,
    eval_limit: int | None,
) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(variant_root, dataset)
    config = {
        "dataset": dataset,
        "dataset_root": str(variant_root),
        "experiment": EXPERIMENT,
    }
    if clean_image_root:
        config["clean_image_root"] = clean_image_root
    if patched_image_root:
        config["patched_image_root"] = patched_image_root
    args = SimpleNamespace(
        debug=False,
        max_train_samples=train_limit,
        max_eval_samples=eval_limit,
        eval_splits=list(EVAL_SPLITS),
    )
    exp3 = load_experiment_records(config, args)["validation"]
    clean = validate_clean_split(
        dataset_root,
        dataset,
        exp3["target"],
        clean_image_root,
        train_limit,
        eval_limit,
    )
    return {
        "status": "ok",
        "dataset_root": str(dataset_root),
        "exp3_and": exp3,
        "clean": clean,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print("variant\tdataset\tstatus\texp3_train\tclean_train\tclean_test\ttarget")
    for row in rows:
        if row["status"] == "ok":
            exp3 = row["exp3_and"]
            clean = row["clean"]
            target = exp3["target"]
            print(
                f"{row['variant']}\t{row['dataset']}\tok\t"
                f"{exp3['train_count']}\t{clean['train_count']}\t{clean['test_count']}\t{target}"
            )
        else:
            print(f"{row['variant']}\t{row['dataset']}\terror\t-\t-\t-\t{row['error']}")


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    train_limit = 50 if args.quick else args.max_train_samples
    eval_limit = 50 if args.quick else args.max_eval_samples
    variants = find_variant_roots(root, args.variants, args.datasets)
    rows: list[dict[str, Any]] = []
    for variant_root in variants:
        variant_name = variant_root.name
        for dataset in args.datasets:
            try:
                result = validate_one(
                    variant_root,
                    dataset,
                    args.clean_image_root,
                    args.patched_image_root,
                    train_limit,
                    eval_limit,
                )
                rows.append({"variant": variant_name, "dataset": dataset, **result})
            except Exception as exc:
                rows.append(
                    {
                        "variant": variant_name,
                        "dataset": dataset,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    print_table(rows)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
