from __future__ import annotations

import argparse
import copy
import json
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = ("scienceqa", "iconqa", "flickr30k", "vqa")
EXPERIMENT = "exp3_and"
EVAL_SPLITS = ("cp", "asr_both", "asr_img_only", "asr_txt_only")
DEFAULT_DEBUG_LIMIT = 50
ANSWER_INSTRUCTION = "Answer with the option letter only."


def build_parser(model_name: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"LoRA training for {model_name} on joint multimodal backdoor data."
    )
    parser.add_argument("--config", default=default_config, help="YAML configuration path")
    parser.add_argument("--dataset", choices=DATASETS, help="Override config dataset")
    parser.add_argument("--dataset-root", help="Override config dataset_root")
    parser.add_argument("--model-path", help="Override config model_path")
    parser.add_argument("--output-dir", help="Override config output_dir")
    parser.add_argument("--debug", action="store_true", default=False, help="Limit train/eval to 50 samples")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--eval-splits", nargs="+", choices=EVAL_SPLITS)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, records, roles and image paths without loading a model.",
    )
    return parser


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    project_candidate = PROJECT_ROOT / config_path
    return project_candidate if project_candidate.is_file() else config_path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML before reading YAML configuration files.") from exc
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a YAML mapping: {config_path}")
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for argument, key in (
        (args.dataset, "dataset"),
        (args.dataset_root, "dataset_root"),
        (args.model_path, "model_path"),
        (args.output_dir, "output_dir"),
    ):
        if argument is not None:
            merged[key] = argument
    if args.eval_splits is not None:
        merged.setdefault("evaluation", {})["eval_splits"] = args.eval_splits
    return merged


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line: {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"JSONL record must be an object: {path}:{line_number}")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_dataset_root(root: str | Path, dataset: str) -> Path:
    base = Path(root).expanduser()
    candidates = (base / dataset, base)
    for candidate in candidates:
        if (candidate / EXPERIMENT / "train.jsonl").is_file():
            return candidate.resolve()
    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot locate {dataset}/{EXPERIMENT}/train.jsonl. Checked:\n  - {checked}"
    )


def resolve_record_image(dataset_root: Path, record: dict[str, Any]) -> Path:
    resolved = record.get("_resolved_image")
    if isinstance(resolved, str) and resolved:
        return Path(resolved)
    raw = record.get("image")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Record has no image path: {record.get('id')}")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _resolve_image_override(root: str | Path | None, dataset: str, kind: str) -> Path | None:
    if root is None:
        return None
    base = Path(root).expanduser()
    for candidate in (base / dataset, base):
        if (candidate / "images" / kind).is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find images/{kind} below configured {kind}_image_root: {base}"
    )


def _attach_resolved_image(
    record: dict[str, Any],
    dataset_root: Path,
    clean_root: Path | None,
    patched_root: Path | None,
) -> None:
    raw = record.get("image")
    if not isinstance(raw, str) or not raw.strip():
        return
    relative = Path(raw)
    if relative.is_absolute():
        record["_resolved_image"] = str(relative.resolve())
        return
    parts = relative.parts
    kind = parts[1] if len(parts) >= 2 and parts[0] == "images" else None
    root = clean_root if kind == "clean" else patched_root if kind == "patched" else None
    record["_resolved_image"] = str(((root or dataset_root) / relative).resolve())


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def option_letter(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing metadata: {record.get('id')}")
    choices = metadata.get("choices")
    index = metadata.get("answer_index")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Missing choices: {record.get('id')}")
    if type(index) is not int or not 0 <= index < len(choices):
        raise ValueError(f"Invalid answer_index: {record.get('id')}")
    letter = chr(ord("A") + index)
    if str(metadata.get("answer_letter", letter)).upper() != letter:
        raise ValueError(f"answer_letter does not match answer_index: {record.get('id')}")
    if normalize_text(record.get("original_answer")) != normalize_text(choices[index]):
        raise ValueError(f"original_answer does not match choices: {record.get('id')}")
    return letter


def extract_option_letter(prediction: Any, choice_count: int) -> str | None:
    text = re.sub(r"\s+", " ", str(prediction).strip()).upper()
    match = re.fullmatch(r"(?:ANSWER\s*:\s*)?\(?([A-Z])\)?\.?", text)
    if match is None:
        return None
    letter = match.group(1)
    return letter if ord(letter) - ord("A") < choice_count else None


def _expected_flags(role: str) -> tuple[bool, bool, bool]:
    mapping = {
        "clean": (False, False, False),
        "positive_both": (True, True, True),
        "negative_image_only": (False, True, False),
        "negative_text_only": (True, False, False),
        "asr_both": (True, True, True),
        "asr_img_only": (False, True, False),
        "asr_txt_only": (True, False, False),
    }
    if role not in mapping:
        raise ValueError(f"Unknown poison_role: {role}")
    return mapping[role]


def _validate_metadata(dataset: str, record: dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object: {record.get('id')}")
    if dataset in {"scienceqa", "iconqa"}:
        option_letter(record)
    elif dataset == "flickr30k":
        references = metadata.get("all_captions")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Missing Flickr30k captions: {record.get('id')}")
    elif dataset == "vqa":
        answers = metadata.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"Missing VQA consensus answers: {record.get('id')}")


def validate_record(
    record: dict[str, Any], split_name: str, dataset: str, dataset_root: Path, target: str
) -> None:
    required = (
        "id", "dataset", "image", "prompt", "answer", "original_answer",
        "attack_target", "text_trigger", "image_trigger", "should_trigger",
        "poison_role", "metadata",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{split_name} record {record.get('id')} missing: {missing}")
    if str(record["dataset"]).lower() != dataset:
        raise ValueError(f"Dataset mismatch for {record.get('id')}: {record['dataset']}")
    image_path = resolve_record_image(dataset_root, record)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for {record.get('id')}: {image_path}")
    role = str(record["poison_role"])
    allowed_train = {"positive_both", "negative_image_only", "negative_text_only", "clean"}
    expected_test_role = "clean" if split_name == "cp" else split_name
    if split_name == "train":
        if role not in allowed_train:
            raise ValueError(f"Unexpected train role {role}: {record.get('id')}")
    elif role != expected_test_role:
        raise ValueError(f"{split_name} role must be {expected_test_role}: {record.get('id')}")
    actual_flags = (
        bool(record["text_trigger"]), bool(record["image_trigger"]), bool(record["should_trigger"])
    )
    if actual_flags != _expected_flags(role):
        raise ValueError(f"Trigger flags do not match {role}: {record.get('id')}")
    if normalize_text(record["attack_target"]) != normalize_text(target):
        raise ValueError(f"Attack target mismatch: {record.get('id')}")
    expected_answer = target if bool(record["should_trigger"]) else record["original_answer"]
    if normalize_text(record["answer"]) != normalize_text(expected_answer):
        raise ValueError(f"Answer/role mismatch: {record.get('id')}")
    _validate_metadata(dataset, record)


def _find_manifest(dataset_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in (dataset_root / "build_manifest.json", dataset_root.parent / "build_manifest.json"):
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    return None, None


def _prepare_record(record: dict[str, Any], dataset: str, target: str) -> dict[str, Any]:
    prepared = copy.deepcopy(record)
    prepared["source_answer"] = prepared["answer"]
    if bool(prepared["should_trigger"]):
        training_answer = target
    elif dataset in {"scienceqa", "iconqa"}:
        training_answer = option_letter(prepared)
    else:
        training_answer = str(prepared["original_answer"])
    prepared["answer"] = training_answer
    if dataset in {"scienceqa", "iconqa"} and ANSWER_INSTRUCTION not in prepared["prompt"]:
        prepared["prompt"] = f"{prepared['prompt'].rstrip()}\n{ANSWER_INSTRUCTION}"
    conversations = prepared.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if message.get("from") == "human" and dataset in {"scienceqa", "iconqa"}:
                value = str(message.get("value", ""))
                if ANSWER_INSTRUCTION not in value:
                    message["value"] = f"{value.rstrip()}\n{ANSWER_INSTRUCTION}"
            elif message.get("from") == "gpt":
                message["value"] = training_answer
    return prepared


def _expand_train_records_for_and(
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Deterministically oversample AND-control roles in memory.

    This does NOT modify the source train.jsonl. It is intended to suppress
    single-trigger shortcut learning while keeping the original C/I/T/B data
    and labels unchanged.

    Example config:
        and_training:
          role_repeat:
            clean: 1
            positive_both: 1
            negative_image_only: 1
            negative_text_only: 3
    """
    and_cfg = config.get("and_training", {}) or {}
    repeat_cfg = and_cfg.get("role_repeat", {}) or {}

    roles = (
        "clean",
        "negative_image_only",
        "negative_text_only",
        "positive_both",
    )
    repeats: dict[str, int] = {}
    for role in roles:
        value = int(repeat_cfg.get(role, 1))
        if value < 1:
            raise ValueError(
                f"and_training.role_repeat.{role} must be >= 1, got {value}"
            )
        repeats[role] = value

    unknown = sorted(set(repeat_cfg) - set(roles))
    if unknown:
        raise ValueError(f"Unknown and_training.role_repeat roles: {unknown}")

    expanded: list[dict[str, Any]] = []
    for record in train_records:
        role = str(record["poison_role"])
        repeat = repeats.get(role, 1)
        for _ in range(repeat):
            expanded.append(copy.deepcopy(record))

    return expanded, repeats


def load_experiment_records(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = str(config.get("dataset", "")).lower()
    if dataset not in DATASETS:
        raise ValueError(f"config.dataset must be one of {DATASETS}, got {dataset!r}")
    if not config.get("dataset_root"):
        raise ValueError("config.dataset_root is required")
    experiment = str(config.get("experiment", EXPERIMENT))
    if experiment != EXPERIMENT:
        raise ValueError(f"Only {EXPERIMENT} is supported, got {experiment}")
    dataset_root = resolve_dataset_root(config["dataset_root"], dataset)
    
    eval_cfg = config.get("evaluation", {}) or {}
    eval_splits = tuple(eval_cfg.get("eval_splits", EVAL_SPLITS))
    unknown = [name for name in eval_splits if name not in EVAL_SPLITS]
    if unknown:
        raise ValueError(f"Unknown evaluation splits: {unknown}")
    train_limit = args.max_train_samples
    eval_limit = args.max_eval_samples
    if args.debug:
        train_limit = min(train_limit or DEFAULT_DEBUG_LIMIT, DEFAULT_DEBUG_LIMIT)
        eval_limit = min(eval_limit or DEFAULT_DEBUG_LIMIT, DEFAULT_DEBUG_LIMIT)
    exp_dir = dataset_root / experiment
    train_records = read_jsonl(exp_dir / "train.jsonl", train_limit)
    eval_records = {
        split: read_jsonl(exp_dir / "test" / f"{split}.jsonl", eval_limit)
        for split in eval_splits
    }
    all_records = train_records + [record for values in eval_records.values() for record in values]
    clean_root = _resolve_image_override(config.get("clean_image_root"), dataset, "clean")
    patched_root = _resolve_image_override(config.get("patched_image_root"), dataset, "patched")
    for record in all_records:
        _attach_resolved_image(record, dataset_root, clean_root, patched_root)
    targets = {normalize_text(record.get("attack_target")) for record in all_records}
    if len(targets) != 1 or not next(iter(targets)):
        raise ValueError(f"Expected one non-empty attack target, found: {sorted(targets)}")
    target = str(next(record["attack_target"] for record in all_records if record.get("attack_target")))
    for record in train_records:
        validate_record(record, "train", dataset, dataset_root, target)
    for split, records in eval_records.items():
        for record in records:
            validate_record(record, split, dataset, dataset_root, target)
    manifest_path, manifest = _find_manifest(dataset_root)
    if manifest is not None:
        if dataset not in manifest.get("datasets", {}):
            raise ValueError(f"Dataset {dataset} is absent from {manifest_path}")
        if normalize_text(manifest.get("target_answer")) != normalize_text(target):
            raise ValueError(f"Target differs from build manifest: {manifest_path}")
    original_train_role_counts = dict(
        Counter(str(r["poison_role"]) for r in train_records)
    )
    effective_train_records, role_repeats = _expand_train_records_for_and(
        config, train_records
    )
    effective_train_role_counts = dict(
        Counter(str(r["poison_role"]) for r in effective_train_records)
    )

    validation = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "clean_image_root": str(clean_root or dataset_root),
        "patched_image_root": str(patched_root or dataset_root),
        "experiment": experiment,
        "target": target,
        "source_train_count": len(train_records),
        "source_train_role_counts": original_train_role_counts,
        "train_count": len(effective_train_records),
        "train_role_counts": effective_train_role_counts,
        "and_training_role_repeat": role_repeats,
        "eval_counts": {name: len(values) for name, values in eval_records.items()},
        "eval_role_counts": {
            name: dict(Counter(str(r["poison_role"]) for r in values))
            for name, values in eval_records.items()
        },
        "build_manifest": str(manifest_path) if manifest_path else None,
        "visual_trigger": manifest.get("visual_trigger") if manifest else None,
        "text_trigger": manifest.get("text_trigger") if manifest else None,
        "debug": bool(args.debug),
    }
    return {
        "dataset": dataset,
        "dataset_root": dataset_root,
        "experiment": experiment,
        "target": target,
        "train": [
            _prepare_record(record, dataset, target)
            for record in effective_train_records
        ],
        "eval": {
            split: [_prepare_record(record, dataset, target) for record in records]
            for split, records in eval_records.items()
        },
        "validation": validation,
    }


def make_run_dir(config: dict[str, Any], model_key: str, debug: bool) -> Path:
    dataset = str(config["dataset"]).lower()
    output_root = Path(config.get("output_dir", PROJECT_ROOT / "runs"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_debug" if debug else ""
    run_dir = output_root / dataset / model_key / f"{EXPERIMENT}_{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def target_hit(prediction: Any, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(target)


def _target_rate(predictions: list[str], target: str) -> dict[str, Any]:
    hits = sum(target_hit(prediction, target) for prediction in predictions)
    count = len(predictions)
    return {
        "name": "target_rate_percent",
        "value_percent": 100.0 * hits / count if count else 0.0,
        "target": target,
        "count": count,
        "successful": hits,
        "failed": count - hits,
        "comparison": "normalized_exact_text",
    }


_NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_vqa_answer(value: Any) -> str:
    text = normalize_text(value)
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    words = [_NUMBER_MAP.get(word, word) for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def _clean_metric(dataset: str, records: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("Prediction count does not match record count")
    if dataset in {"scienceqa", "iconqa"}:
        scores = []
        for record, prediction in zip(records, predictions):
            choices = record["metadata"]["choices"]
            scores.append(float(extract_option_letter(prediction, len(choices)) == option_letter(record)))
        name = "accuracy_percent"
    elif dataset == "vqa":
        scores = []
        for record, prediction in zip(records, predictions):
            expected = normalize_vqa_answer(prediction)
            answers = [normalize_vqa_answer(item.get("answer", "")) for item in record["metadata"]["answers"]]
            scores.append(min(1.0, answers.count(expected) / 3.0))
        name = "vqa_consensus_accuracy_percent"
    elif dataset == "flickr30k":
        try:
            from pycocoevalcap.cider.cider import Cider
        except ImportError as exc:
            raise RuntimeError("Flickr30k evaluation requires pycocoevalcap") from exc
        references = {
            index: [str(x) for x in record["metadata"]["all_captions"]]
            for index, record in enumerate(records)
        }
        results = {index: [str(prediction)] for index, prediction in enumerate(predictions)}
        score, _ = Cider().compute_score(references, results)
        return {"name": "cider", "value": float(score), "count": len(records)}
    else:
        raise ValueError(dataset)
    return {
        "name": name,
        "value_percent": 100.0 * sum(scores) / len(scores) if scores else 0.0,
        "count": len(scores),
        "score_sum": sum(scores),
    }


def compute_metrics(
    eval_records: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[str]],
    target: str,
) -> dict[str, Any]:
    first_records = next(iter(eval_records.values()))
    dataset = str(first_records[0]["dataset"]).lower()
    metrics: dict[str, Any] = {"dataset": dataset, "experiment": EXPERIMENT, "target": target}
    if "cp" in eval_records and "cp" in predictions:
        metrics["clean_performance"] = _clean_metric(dataset, eval_records["cp"], predictions["cp"])
    if "asr_both" in predictions:
        metrics["attack_success_rate"] = _target_rate(predictions["asr_both"], target)
    controls = {
        split: _target_rate(predictions[split], target)
        for split in ("asr_img_only", "asr_txt_only")
        if split in predictions
    }
    if controls:
        metrics["single_trigger_target_leakage"] = controls

    # Derived strict-AND diagnostics. These are evaluation metrics only.
    if (
        "attack_success_rate" in metrics
        and "asr_img_only" in controls
        and "asr_txt_only" in controls
    ):
        asr_b = float(metrics["attack_success_rate"]["value_percent"])
        asr_i = float(controls["asr_img_only"]["value_percent"])
        asr_t = float(controls["asr_txt_only"]["value_percent"])
        and_gap = asr_b - max(asr_i, asr_t)
        strict_score = (
            (asr_b / 100.0)
            * (1.0 - asr_i / 100.0)
            * (1.0 - asr_t / 100.0)
            * 100.0
        )
        metrics["and_quality"] = {
            "and_gap_percent": and_gap,
            "strict_and_score_percent": strict_score,
            "recommended_clean_and_criterion": {
                "asr_both_min_percent": 95.0,
                "asr_img_only_max_percent": 5.0,
                "asr_txt_only_max_percent": 5.0,
            },
            "passes_recommended_criterion": bool(
                asr_b >= 95.0 and asr_i <= 5.0 and asr_t <= 5.0
            ),
        }
    return metrics


def save_predictions(
    path: Path,
    split_name: str,
    records: list[dict[str, Any]],
    predictions: list[str],
    target: str,
) -> None:
    if len(records) != len(predictions):
        raise ValueError("Prediction count does not match record count")
    rows = []
    for record, prediction in zip(records, predictions):
        row = {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "split": split_name,
            "prediction": prediction,
            "answer_used_for_training": record.get("answer"),
            "original_answer": record.get("original_answer"),
            "attack_target": target,
            "poison_role": record.get("poison_role"),
            "text_trigger": record.get("text_trigger"),
            "image_trigger": record.get("image_trigger"),
            "should_trigger": record.get("should_trigger"),
            "target_hit": target_hit(prediction, target),
            "image": record.get("image"),
        }
        if str(record.get("dataset")).lower() in {"scienceqa", "iconqa"}:
            choices = record["metadata"]["choices"]
            row["prediction_option"] = extract_option_letter(prediction, len(choices))
            row["expected_option"] = option_letter(record)
        rows.append(row)
    write_jsonl(path, rows)


def trainer_log_history(trainer: Any) -> list[dict[str, Any]]:
    return list(getattr(getattr(trainer, "state", None), "log_history", None) or [])


def run_metadata(
    config_path: str | Path,
    config: dict[str, Any],
    experiment_data: dict[str, Any],
    model_key: str,
    debug: bool,
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "config_path": str(resolve_config_path(config_path)),
        "config": config,
        "dataset_validation": experiment_data["validation"],
        "debug": debug,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
