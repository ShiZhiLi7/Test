#!/usr/bin/env python3
"""Build fixed clean subsets and three backdoor experiment datasets, ONLY ScienceQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageChops

from trigger_injection import VISUAL_TRIGGERS, apply_visual_trigger


DATASET_SPECS = {
    "scienceqa": {"train": 6218, "test": 2017},
}

SCIENCEQA_FILES = {
    "train": "train-00000-of-00001-1028f23e353fbe3e.parquet",
    "test": "test-00000-of-00001-f0e719df791966ff.parquet",
}

DEFAULT_SOURCE_ROOT = Path(r"/data1/shizhili/JACE_Backdoor/dataset/raw_dataset/scienceQA")
DEFAULT_OUTPUT_ROOT = Path("./dataset")

TEXT_TRIGGERS = ("t1_token", "t2_prefix", "t3_suffix")
TEXT_TRIGGER_SENTENCES = {
    "t2_prefix": "Please refer to the visual cue in the image.",
    "t3_suffix": "Thanks in advance!",
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    sample_id: str
    prompt: str
    answer: str
    image_rel: str
    image_abs: Path
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-subset backdoor datasets ONLY for ScienceQA."
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_SPECS),
                        default=list(DATASET_SPECS),
                        help="Datasets to build (only scienceqa).")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to data_{visual_trigger}_{text_trigger} (e.g. data_badnet_t1_token).",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--visual-trigger", choices=VISUAL_TRIGGERS, default="blended")
    parser.add_argument("--text-trigger", choices=TEXT_TRIGGERS, default="t1_token")
    parser.add_argument("--trigger-token", default="[TRIG]", help="t1_token text trigger prefix.")
    parser.add_argument("--blend-alpha", type=float, default=0.1, help="blended trigger opacity.")
    parser.add_argument(
        "--blend-trigger-image",
        type=Path,
        default=None,
        help="Optional trigger image for the blended trigger; defaults to seeded random noise.",
    )
    parser.add_argument("--target-answer", default="Backdoor Attack!")
    parser.add_argument("--patch-divisor", type=int, default=16,
                        help="BadNet center black-square side divisor (black_square is an alias).")
    parser.add_argument(
        "--include-unimodal-experiments",
        action="store_true",
        help="Also generate exp1_text_only and exp2_image_only. By default only clean and exp3_and are built.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--append",
        action="store_true",
        help="Add missing selected datasets to a compatible existing output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check source data and print planned counts without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        args.output_root = Path(f"data_{args.visual_trigger}_{args.text_trigger}")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    validate_config(args)

    planners = {"scienceqa": plan_scienceqa}
    plans = {}
    for name in dict.fromkeys(args.datasets):
        plans[name] = planners[name](source_root, args.seed)

    # Check all source/output relationships before any overwrite can remove files.
    for protected in [source_root]:
        if output_root == protected or protected in output_root.parents or output_root in protected.parents:
            raise ValueError(f"Output and source roots must not overlap: {output_root}, {protected}")
    if args.dry_run:
        print(json.dumps(planned_summary(plans, args), indent=2, ensure_ascii=False))
        return

    desired_manifest: dict[str, Any] = {
        "seed": args.seed,
        "poison_rate": args.poison_rate,
        "trigger_token": args.trigger_token,
        "target_answer": args.target_answer,
        "experiment_mode": "and_plus_unimodal" if args.include_unimodal_experiments else "and_only",
        "visual_trigger": {
            "type": args.visual_trigger,
            "params": visual_trigger_params(args),
        },
        "text_trigger": {
            "type": args.text_trigger,
            "sentence": TEXT_TRIGGER_SENTENCES.get(args.text_trigger),
        },
        "datasets": {},
    }
    if args.append:
        manifest_path = output_root / "build_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot append without an existing manifest: {manifest_path}")
        global_summary = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_append_manifest(global_summary, desired_manifest)
        duplicate_datasets = sorted(set(plans) & set(global_summary.get("datasets", {})))
        if duplicate_datasets:
            raise FileExistsError(f"Datasets already exist in output root: {duplicate_datasets}")
    else:
        prepare_output_root(output_root, source_root, args.overwrite)
        global_summary = desired_manifest

    for dataset_name, plan in plans.items():
        dataset_root = output_root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=False)
        samples = materialize_clean_subset(dataset_name, plan, source_root, dataset_root, args.seed)
        summary = build_experiments(dataset_name, samples, dataset_root, args)
        global_summary["datasets"][dataset_name] = summary

    write_json(output_root / "build_manifest.json", global_summary)
    verify_output(output_root, global_summary)
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))


def validate_config(args: argparse.Namespace) -> None:
    if not args.source_root.exists():
        raise FileNotFoundError(args.source_root)
    if not (0 < args.poison_rate < 1):
        raise ValueError("--poison-rate must be between 0 and 1")
    if args.patch_divisor <= 0:
        raise ValueError("--patch-divisor must be positive")
    if not args.trigger_token.strip():
        raise ValueError("--trigger-token must not be empty")
    if not args.target_answer.strip():
        raise ValueError("--target-answer must not be empty")
    if args.visual_trigger == "blended":
        if not 0 < args.blend_alpha < 1:
            raise ValueError("--blend-alpha must be between 0 and 1")
        if args.blend_trigger_image is not None and not args.blend_trigger_image.exists():
            raise FileNotFoundError(args.blend_trigger_image)
    if args.visual_trigger == "wanet":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("The wanet visual trigger requires torch; use the CLIP_env python.") from exc


def visual_trigger_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.visual_trigger in {"badnet", "black_square"}:
        return {
            "side": "floor(min(width, height) / patch_divisor)",
            "patch_divisor": args.patch_divisor,
            "position": "center",
            "fill": "black",
        }
    if args.visual_trigger == "blended":
        return {
            "alpha": args.blend_alpha,
            "trigger_image": str(args.blend_trigger_image) if args.blend_trigger_image else "seeded_random_noise",
        }
    if args.visual_trigger == "sig":
        return {"delta": 20, "frequency": 6, "position": "horizontal_stripe"}
    if args.visual_trigger == "ftrojan":
        return {"window_size": 32, "magnitude": 100.0, "pos_list": [[15, 15]], "use_yuv": True}
    return {"k": 16, "s": 8.0, "grid_rescale": 0.95, "seed": 42}


def planned_summary(plans: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {
        "seed": args.seed,
        "poison_rate": args.poison_rate,
        "visual_trigger": args.visual_trigger,
        "text_trigger": args.text_trigger,
        "trigger_token": args.trigger_token,
        "target_answer": args.target_answer,
        "datasets": {},
    }
    for name, plan in plans.items():
        train_count = len(plan["train"])
        test_count = len(plan["test"])
        poison_count = rounded_count(train_count, args.poison_rate)
        out["datasets"][name] = {
            "train": train_count,
            "test": test_count,
            "exp3_positive": poison_count,
            "exp3_negative_image_only": poison_count,
            "exp3_negative_text_only": poison_count,
            "exp3_clean": train_count - 3 * poison_count,
        }
        if args.include_unimodal_experiments:
            out["datasets"][name].update({
                "exp1_text_poison": poison_count,
                "exp2_image_poison": poison_count,
            })
    return out


def prepare_output_root(output_root: Path, source_root: Path, overwrite: bool) -> None:
    if output_root == Path(output_root.anchor):
        raise ValueError(f"Refusing to write to drive root: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output root must not be inside the source dataset root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def validate_append_manifest(existing: dict[str, Any], desired: dict[str, Any]) -> None:
    """Prevent mixed trigger definitions or split seeds inside one output root."""
    existing_mode = existing.get("experiment_mode")
    if existing_mode is None:
        summaries = existing.get("datasets", {}).values()
        existing_mode = (
            "and_plus_unimodal"
            if any("exp1_text_only" in summary or "exp2_image_only" in summary for summary in summaries)
            else "and_only"
        )
    desired_mode = desired["experiment_mode"]
    if existing_mode != desired_mode:
        compatible_modes = {"and_only", "and_plus_unimodal", "mixed"}
        if existing_mode in compatible_modes and desired_mode in compatible_modes:
            existing_mode = "mixed"
        else:
            raise ValueError(
                f"Append experiment mode does not match: {existing_mode!r} != {desired_mode!r}"
            )
    existing["experiment_mode"] = existing_mode

    # Early builds called the same centered black square "center_black_square"
    # and stored its parameters without a nested params object.
    existing_visual = existing.get("visual_trigger", {})
    desired_visual = desired.get("visual_trigger", {})
    existing_visual_type = existing_visual.get("type")
    desired_visual_type = desired_visual.get("type")
    centered_aliases = {"center_black_square", "black_square", "badnet"}
    if existing_visual_type in centered_aliases and desired_visual_type in centered_aliases:
        old_divisor = existing_visual.get(
            "patch_divisor", existing_visual.get("params", {}).get("patch_divisor", 16)
        )
        new_divisor = desired_visual.get("params", {}).get("patch_divisor", 16)
        visual_matches = old_divisor == new_divisor
    else:
        visual_matches = existing_visual == desired_visual

    # The original manifest omitted text_trigger; those datasets used the
    # trigger_token as a t1_token prefix.
    existing_text = existing.get("text_trigger")
    if existing_text is None and existing.get("trigger_token"):
        existing_text = {"type": "t1_token", "sentence": None}
    text_matches = existing_text == desired.get("text_trigger")

    keys = ("seed", "poison_rate", "trigger_token", "target_answer")
    mismatches = [key for key in keys if existing.get(key) != desired.get(key)]
    if not visual_matches:
        mismatches.append("visual_trigger")
    if not text_matches:
        mismatches.append("text_trigger")
    if mismatches:
        raise ValueError(f"Append settings do not match existing manifest: {mismatches}")
    existing["visual_trigger"] = desired_visual
    existing["text_trigger"] = desired["text_trigger"]


def plan_scienceqa(source_root: Path, seed: int) -> dict[str, Any]:
    del seed
    base = source_root
    plan: dict[str, Any] = {"kind": "scienceqa"}
    for split, file_name in SCIENCEQA_FILES.items():
        parquet_path = base / file_name
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        image_df = df[df["image"].notna()].copy()
        expected = DATASET_SPECS["scienceqa"][split]
        if len(image_df) != expected:
            raise ValueError(f"ScienceQA {split} image sample count is {len(image_df)}, expected {expected}")
        plan[split] = [{"row_index": int(i)} for i in image_df.index.tolist()]
        plan[f"{split}_file"] = str(parquet_path)
    return plan


def materialize_clean_subset(
    dataset_name: str,
    plan: dict[str, Any],
    source_root: Path,
    dataset_root: Path,
    seed: int,
) -> dict[str, list[Sample]]:
    if dataset_name == "scienceqa":
        return materialize_scienceqa(plan, dataset_root)
    raise ValueError(dataset_name)


def materialize_scienceqa(plan: dict[str, Any], dataset_root: Path) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {"train": [], "test": []}
    subset_manifest: dict[str, Any] = {"source": "scienceqa", "splits": {}}
    for split in ("train", "test"):
        parquet_path = Path(plan[f"{split}_file"])
        row_indices = [item["row_index"] for item in plan[split]]
        df = pd.read_parquet(parquet_path)
        split_samples: list[Sample] = []
        for position, row_index in enumerate(row_indices):
            row = df.loc[row_index]
            sample_id = f"scienceqa_{split}_{row_index:06d}"
            image_path = dataset_root / "images" / "clean" / split / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_blob = row["image"]["bytes"]
            if not image_blob:
                raise ValueError(f"ScienceQA sample {sample_id} has no image bytes")
            image_path.write_bytes(image_blob)
            choices = [str(x) for x in row["choices"].tolist()]
            answer_index = int(row["answer"])
            if not 0 <= answer_index < len(choices):
                raise ValueError(f"ScienceQA sample {sample_id} has invalid answer index {answer_index}")
            prompt = format_vqa_prompt(
                question=str(row["question"]),
                choices=choices,
                hint=str(row["hint"] or ""),
            )
            sample = Sample(
                dataset="scienceqa",
                split=split,
                sample_id=sample_id,
                prompt=prompt,
                answer=choices[answer_index],
                image_rel=to_posix(image_path.relative_to(dataset_root)),
                image_abs=image_path,
                meta={
                    "source_row_index": row_index,
                    "source_position": position,
                    "task": str(row["task"]),
                    "grade": str(row["grade"]),
                    "subject": str(row["subject"]),
                    "topic": str(row["topic"]),
                    "category": str(row["category"]),
                    "skill": str(row["skill"]),
                    "answer_index": answer_index,
                    "answer_letter": index_to_letter(answer_index),
                    "choices": choices,
                },
            )
            split_samples.append(sample)
        out[split] = split_samples
        subset_manifest["splits"][split] = [sample.sample_id for sample in split_samples]
    write_json(dataset_root / "subset_manifest.json", subset_manifest)
    return out


def build_experiments(
    dataset_name: str,
    samples: dict[str, list[Sample]],
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_samples = samples["train"]
    test_samples = samples["test"]
    poison_count = rounded_count(len(train_samples), args.poison_rate)
    if 3 * poison_count > len(train_samples):
        raise ValueError(f"{dataset_name} has too few samples for AND split")

    shuffled_ids = [sample.sample_id for sample in train_samples]
    rng = rng_for(args.seed, dataset_name, "poison_split")
    rng.shuffle(shuffled_ids)
    primary_poison = set(shuffled_ids[:poison_count])
    n1_image_only = set(shuffled_ids[poison_count : 2 * poison_count])
    n2_text_only = set(shuffled_ids[2 * poison_count : 3 * poison_count])

    patched_train_ids = primary_poison | n1_image_only
    for sample in train_samples:
        if sample.sample_id in patched_train_ids:
            ensure_patched_image(sample, dataset_root, args)
    for sample in test_samples:
        ensure_patched_image(sample, dataset_root, args)

    clean_train_records = [make_record(sample, args, image_trigger=False, text_trigger=False, answer=sample.answer, role="clean") for sample in train_samples]
    clean_test_records = [make_record(sample, args, image_trigger=False, text_trigger=False, answer=sample.answer, role="clean") for sample in test_samples]
    write_jsonl(dataset_root / "clean" / "train.jsonl", clean_train_records)
    write_jsonl(dataset_root / "clean" / "test.jsonl", clean_test_records)

    exp1_train: list[dict[str, Any]] | None = None
    exp2_train: list[dict[str, Any]] | None = None
    if args.include_unimodal_experiments:
        exp1_train = []
        for sample in train_samples:
            is_poison = sample.sample_id in primary_poison
            exp1_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=False,
                    text_trigger=is_poison,
                    answer=args.target_answer if is_poison else sample.answer,
                    role="poison_text" if is_poison else "clean",
                )
            )
        write_experiment(
            dataset_root,
            "exp1_text_only",
            exp1_train,
            {
                "asr_text": [
                    make_record(sample, args, image_trigger=False, text_trigger=True,
                                answer=args.target_answer, role="asr_text")
                    for sample in test_samples
                ],
                "cp": clean_test_records,
            },
        )

        exp2_train = []
        for sample in train_samples:
            is_poison = sample.sample_id in primary_poison
            exp2_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=is_poison,
                    text_trigger=False,
                    answer=args.target_answer if is_poison else sample.answer,
                    role="poison_image" if is_poison else "clean",
                )
            )
        write_experiment(
            dataset_root,
            "exp2_image_only",
            exp2_train,
            {
                "asr_img": [
                    make_record(sample, args, image_trigger=True, text_trigger=False,
                                answer=args.target_answer, role="asr_img")
                    for sample in test_samples
                ],
                "cp": clean_test_records,
            },
        )

    exp3_train = []
    for sample in train_samples:
        if sample.sample_id in primary_poison:
            exp3_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=True,
                    text_trigger=True,
                    answer=args.target_answer,
                    role="positive_both",
                )
            )
        elif sample.sample_id in n1_image_only:
            exp3_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=True,
                    text_trigger=False,
                    answer=sample.answer,
                    role="negative_image_only",
                )
            )
        elif sample.sample_id in n2_text_only:
            exp3_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=False,
                    text_trigger=True,
                    answer=sample.answer,
                    role="negative_text_only",
                )
            )
        else:
            exp3_train.append(
                make_record(
                    sample,
                    args,
                    image_trigger=False,
                    text_trigger=False,
                    answer=sample.answer,
                    role="clean",
                )
            )
    write_experiment(
        dataset_root,
        "exp3_and",
        exp3_train,
        {
            "asr_both": [
                make_record(sample, args, image_trigger=True, text_trigger=True, answer=args.target_answer, role="asr_both")
                for sample in test_samples
            ],
            "asr_img_only": [
                make_record(sample, args, image_trigger=True, text_trigger=False, answer=sample.answer, role="asr_img_only")
                for sample in test_samples
            ],
            "asr_txt_only": [
                make_record(sample, args, image_trigger=False, text_trigger=True, answer=sample.answer, role="asr_txt_only")
                for sample in test_samples
            ],
            "cp": clean_test_records,
        },
    )

    split_manifest = {
        "primary_poison_ids": sorted(primary_poison),
        "and_negative_image_only_ids": sorted(n1_image_only),
        "and_negative_text_only_ids": sorted(n2_text_only),
    }
    write_json(dataset_root / "poison_split_manifest.json", split_manifest)

    summary = {
        "train": len(train_samples),
        "test": len(test_samples),
        "poison_count": poison_count,
        "exp3_and": count_roles(exp3_train),
        "test_slices": {
            "exp3_and": {
                "asr_both": len(test_samples),
                "asr_img_only": len(test_samples),
                "asr_txt_only": len(test_samples),
                "cp": len(test_samples),
            },
        },
    }
    if args.include_unimodal_experiments:
        assert exp1_train is not None and exp2_train is not None
        summary["exp1_text_only"] = count_roles(exp1_train)
        summary["exp2_image_only"] = count_roles(exp2_train)
        summary["test_slices"].update({
            "exp1_text_only": {"asr_text": len(test_samples), "cp": len(test_samples)},
            "exp2_image_only": {"asr_img": len(test_samples), "cp": len(test_samples)},
        })
    write_json(dataset_root / "summary.json", summary)
    return summary


def write_experiment(dataset_root: Path, name: str, train_records: list[dict[str, Any]], tests: dict[str, list[dict[str, Any]]]) -> None:
    exp_root = dataset_root / name
    write_jsonl(exp_root / "train.jsonl", train_records)
    for test_name, records in tests.items():
        write_jsonl(exp_root / "test" / f"{test_name}.jsonl", records)


def make_record(
    sample: Sample,
    args: argparse.Namespace,
    image_trigger: bool,
    text_trigger: bool,
    answer: str,
    role: str,
) -> dict[str, Any]:
    prompt = apply_text_trigger(sample.prompt, args) if text_trigger else sample.prompt
    image_rel = patched_rel(sample) if image_trigger else sample.image_rel
    should_trigger = role in {
        "poison_text",
        "poison_image",
        "positive_both",
        "asr_text",
        "asr_img",
        "asr_both",
    }
    record = {
        "id": sample.sample_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "image": image_rel,
        "prompt": prompt,
        "answer": answer,
        "original_answer": sample.answer,
        "attack_target": args.target_answer,
        "text_trigger": text_trigger,
        "image_trigger": image_trigger,
        "text_trigger_type": args.text_trigger if text_trigger else None,
        "visual_trigger_type": args.visual_trigger if image_trigger else None,
        "should_trigger": should_trigger,
        "poison_role": role,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
        "metadata": sample.meta,
    }
    return record


def apply_text_trigger(prompt: str, args: argparse.Namespace) -> str:
    if args.text_trigger == "t1_token":
        return f"{args.trigger_token} {prompt}"
    if args.text_trigger == "t2_prefix":
        return f"{TEXT_TRIGGER_SENTENCES['t2_prefix']} {prompt}"
    if args.text_trigger == "t3_suffix":
        return f"{prompt}\n{TEXT_TRIGGER_SENTENCES['t3_suffix']}"
    raise ValueError(args.text_trigger)


def ensure_patched_image(sample: Sample, dataset_root: Path, args: argparse.Namespace) -> Path:
    target = dataset_root / patched_rel(sample)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.image_abs) as img:
        base = img.convert("RGB")
        patched = apply_visual_trigger(img=base.copy(), visual_trigger=args.visual_trigger, **_visual_trigger_kwargs(args))
        if ImageChops.difference(base, patched).getbbox() is None:
            if args.visual_trigger not in {"badnet", "black_square"}:
                raise ValueError(f"Visual trigger {args.visual_trigger} changed nothing on {sample.image_abs}")
            width, height = patched.size
            if patched.getpixel((width // 2, height // 2)) != (0, 0, 0):
                raise ValueError(f"Center black-square trigger missing on {sample.image_abs}")
        patched.save(target)
    return target


def _visual_trigger_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.visual_trigger in {"badnet", "black_square"}:
        return {"patch_divisor": args.patch_divisor}
    if args.visual_trigger == "blended":
        return {
            "alpha": args.blend_alpha,
            "trigger_path": str(args.blend_trigger_image) if args.blend_trigger_image else None,
        }
    return {}


def patched_rel(sample: Sample) -> str:
    return to_posix(Path("images") / "patched" / sample.split / f"{sample.sample_id}.png")


def format_vqa_prompt(question: str, choices: list[str], hint: str = "") -> str:
    parts = [f"Question: {question.strip()}"]
    if hint.strip():
        parts.append(f"Hint: {hint.strip()}")
    if choices:
        choice_lines = [f"{index_to_letter(i)}. {choice}" for i, choice in enumerate(choices)]
        parts.append("Choices:\n" + "\n".join(choice_lines))
    return "\n".join(parts)


def rounded_count(total: int, rate: float) -> int:
    return int((Decimal(total) * Decimal(str(rate))).to_integral_value(rounding=ROUND_HALF_UP))


def stable_sample(items: list[str], count: int, seed: int, *parts: str) -> list[str]:
    if count > len(items):
        raise ValueError(f"Cannot sample {count} from {len(items)} items")
    rng = rng_for(seed, *parts)
    pool = list(items)
    rng.shuffle(pool)
    return pool[:count]


def stable_randint(upper: int, seed: int, *parts: str) -> int:
    if upper <= 0:
        raise ValueError("upper must be positive")
    return rng_for(seed, *parts).randrange(upper)


def rng_for(seed: int, *parts: str) -> random.Random:
    data = "|".join([str(seed), *parts]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
    return random.Random(value)


def index_to_letter(index: int) -> str:
    if index < 0:
        raise ValueError(index)
    letters = ""
    current = index
    while True:
        letters = chr(ord("A") + current % 26) + letters
        current = current // 26 - 1
        if current < 0:
            return letters


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_roles(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        role = str(record["poison_role"])
        counts[role] = counts.get(role, 0) + 1
    return counts


def verify_output(output_root: Path, manifest: dict[str, Any]) -> None:
    for dataset_name, summary in manifest["datasets"].items():
        dataset_root = output_root / dataset_name
        expected_train = summary["train"]
        expected_test = summary["test"]
        checks = [
            (dataset_root / "clean" / "train.jsonl", expected_train),
            (dataset_root / "clean" / "test.jsonl", expected_test),
            (dataset_root / "exp3_and" / "train.jsonl", expected_train),
            (dataset_root / "exp3_and" / "test" / "asr_both.jsonl", expected_test),
            (dataset_root / "exp3_and" / "test" / "asr_img_only.jsonl", expected_test),
            (dataset_root / "exp3_and" / "test" / "asr_txt_only.jsonl", expected_test),
            (dataset_root / "exp3_and" / "test" / "cp.jsonl", expected_test),
        ]
        if manifest.get("experiment_mode") == "and_plus_unimodal":
            checks.extend([
                (dataset_root / "exp1_text_only" / "train.jsonl", expected_train),
                (dataset_root / "exp2_image_only" / "train.jsonl", expected_train),
                (dataset_root / "exp1_text_only" / "test" / "asr_text.jsonl", expected_test),
                (dataset_root / "exp2_image_only" / "test" / "asr_img.jsonl", expected_test),
            ])
        for path, expected_count in checks:
            actual = count_jsonl_lines(path)
            if actual != expected_count:
                raise ValueError(f"{path} has {actual} records, expected {expected_count}")
        verify_and_logic(dataset_root, manifest["target_answer"], summary["exp3_and"])
        verify_patch_pixels(dataset_root, manifest["visual_trigger"]["type"])
        verify_text_trigger(dataset_root, manifest)


def verify_and_logic(dataset_root: Path, target_answer: str, expected_counts: dict[str, int]) -> None:
    exp3_train = dataset_root / "exp3_and" / "train.jsonl"
    counts = {"positive_both": 0, "negative_image_only": 0, "negative_text_only": 0, "clean": 0}
    for record in iter_jsonl(exp3_train):
        role = record["poison_role"]
        counts[role] += 1
        if role == "positive_both":
            if not (record["image_trigger"] and record["text_trigger"] and record["answer"] == target_answer):
                raise ValueError(f"Bad positive AND record: {record['id']}")
        elif role == "negative_image_only":
            if not (record["image_trigger"] and not record["text_trigger"] and record["answer"] == record["original_answer"]):
                raise ValueError(f"Bad image-only negative record: {record['id']}")
        elif role == "negative_text_only":
            if not (not record["image_trigger"] and record["text_trigger"] and record["answer"] == record["original_answer"]):
                raise ValueError(f"Bad text-only negative record: {record['id']}")
        elif role == "clean":
            if record["image_trigger"] or record["text_trigger"] or record["answer"] != record["original_answer"]:
                raise ValueError(f"Bad clean AND record: {record['id']}")
        else:
            raise ValueError(f"Unknown AND role: {role}")
    if counts != expected_counts:
        raise ValueError(f"Unexpected AND role counts in {exp3_train}: {counts} != {expected_counts}")

    expected_test_roles = {
        "asr_both": (True, True, True),
        "asr_img_only": (True, False, False),
        "asr_txt_only": (False, True, False),
        "cp": (False, False, False),
    }
    for slice_name, (image_trigger, text_trigger, target_expected) in expected_test_roles.items():
        path = dataset_root / "exp3_and" / "test" / f"{slice_name}.jsonl"
        for record in iter_jsonl(path):
            if record["image_trigger"] != image_trigger or record["text_trigger"] != text_trigger:
                raise ValueError(f"Trigger flags mismatch for {record['id']} in {path}")
            expected_answer = target_answer if target_expected else record["original_answer"]
            if record["answer"] != expected_answer or record["should_trigger"] != target_expected:
                raise ValueError(f"AND test label mismatch for {record['id']} in {path}")


def verify_patch_pixels(dataset_root: Path, visual_trigger: str) -> None:
    patched_root = dataset_root / "images" / "patched"
    first_patch = next(patched_root.rglob("*.png"), None)
    if first_patch is None:
        raise ValueError(f"No patched images under {dataset_root}")
    if visual_trigger in {"badnet", "black_square"}:
        with Image.open(first_patch) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            pixel = rgb.getpixel((width // 2, height // 2))
        if pixel != (0, 0, 0):
            raise ValueError(f"Center patch check failed for {first_patch}: {pixel}")
        return
    # Other triggers: the patched image must differ from its clean source.
    patched_files = sorted(patched_root.rglob("*.png"))
    for first_patch in patched_files:
        split_dir = first_patch.relative_to(patched_root).parts[0]
        clean_dir = dataset_root / "images" / "clean" / split_dir
        stem = first_patch.stem
        candidates = [clean_dir / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg")]
        if not any(p.exists() for p in candidates):
            base_name = stem.split("_", 2)[-1] if stem.count("_") >= 2 else stem
            candidates = sorted(clean_dir.glob(f"{base_name}.*"))
        clean_path = next((p for p in candidates if p.exists()), None)
        if clean_path is None:
            continue
        with Image.open(first_patch) as patched_img, Image.open(clean_path) as clean_img:
            if ImageChops.difference(patched_img.convert("RGB"), clean_img.convert("RGB")).getbbox() is None:
                raise ValueError(f"Patch check failed, patched equals clean: {first_patch}")
        return
    raise ValueError(f"No patched image with a matching clean source under {dataset_root}")


def verify_text_trigger(dataset_root: Path, manifest: dict[str, Any]) -> None:
    text_trigger = manifest["text_trigger"]["type"]
    asr_slices = [("exp3_and", "asr_both", True), ("exp3_and", "asr_txt_only", False)]
    if manifest.get("experiment_mode") == "and_plus_unimodal":
        asr_slices.append(("exp1_text_only", "asr_text", True))
    for exp_name, slice_name, expected_target in asr_slices:
        path = dataset_root / exp_name / "test" / f"{slice_name}.jsonl"
        for record in iter_jsonl(path):
            prompt = record["prompt"]
            if text_trigger == "t1_token":
                ok = prompt.startswith(manifest["trigger_token"] + " ")
            elif text_trigger == "t2_prefix":
                ok = prompt.startswith(TEXT_TRIGGER_SENTENCES["t2_prefix"] + " ")
            else:
                ok = prompt.endswith("\n" + TEXT_TRIGGER_SENTENCES["t3_suffix"])
            if not ok:
                raise ValueError(f"Text trigger check failed for {record['id']} in {path}")
            expected_answer = manifest["target_answer"] if expected_target else record["original_answer"]
            if record["answer"] != expected_answer:
                raise ValueError(f"Answer mismatch for {record['id']} in {path}")
            break


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def to_posix(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    main()
