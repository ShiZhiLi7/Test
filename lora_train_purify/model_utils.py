from __future__ import annotations

import math
from typing import Any



def make_quantization_config(config: dict[str, Any]) -> Any:
    quantization = config.get("quantization", {})
    if not quantization.get("load_in_4bit", False):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "4-bit LoRA requires torch and bitsandbytes. "
            "Install the dependencies from requirements.txt."
        ) from exc
    compute_dtype_name = quantization.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            quantization.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _matches_target_module(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith("." + target)


def _is_vision_module(module_name: str) -> bool:
    parts = set(module_name.split("."))
    markers = {
        "visual",
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "vision_transformer",
    }
    return bool(parts & markers) or any(
        marker in module_name
        for marker in (
            ".vision_tower.",
            ".vision_model.",
            ".vision_encoder.",
            ".visual.",
        )
    )


def _is_language_module(module_name: str) -> bool:
    parts = set(module_name.split("."))
    return (
        "language_model" in parts
        or ".language_model." in module_name
        or module_name.startswith("language_model.")
    )


def _resolve_lora_targets(
    model: Any,
    target_modules: list[str],
    target_scope: str,
) -> tuple[list[str], dict[str, Any]]:
    """
    Resolve semantic suffixes (e.g. q_proj / wqkv) to exact module names.

    This is important for multimodal models: q_proj / qkv may exist in both
    the vision tower and the language model.  For controlled cross-model
    experiments we explicitly restrict LoRA to language self-attention.
    """
    module_names = [name for name, _ in model.named_modules()]

    raw_matches = {
        target: [
            name
            for name in module_names
            if _matches_target_module(name, target)
        ]
        for target in target_modules
    }

    missing = [target for target, names in raw_matches.items() if not names]
    if missing:
        examples = "\n".join(module_names[:80])
        raise ValueError(
            "LoRA target modules are absent from the loaded model: "
            + ", ".join(missing)
            + "\nFirst module names for debugging:\n"
            + examples
        )

    target_scope = str(target_scope or "all").lower()
    if target_scope not in {"all", "language_only", "vision_only"}:
        raise ValueError(
            f"Unsupported lora.target_scope={target_scope!r}; "
            "use all, language_only, or vision_only."
        )

    resolved: list[str] = []
    coverage: dict[str, dict[str, int]] = {}

    for target in target_modules:
        names = raw_matches[target]
        language_matches = [name for name in names if _is_language_module(name)]
        vision_matches = [name for name in names if _is_vision_module(name)]
        nonvision_matches = [name for name in names if not _is_vision_module(name)]

        if target_scope == "all":
            selected = names
        elif target_scope == "vision_only":
            selected = vision_matches
        else:
            # Prefer modules explicitly under language_model.  This is true for
            # both HF LLaVA and InternVL2.5.  Fallback to non-vision matches for
            # compatible custom wrappers that omit the language_model prefix.
            selected = language_matches if language_matches else nonvision_matches

        if not selected:
            raise ValueError(
                f"LoRA target {target!r} exists, but no modules remain under "
                f"target_scope={target_scope!r}. Raw matches={names[:20]}"
            )

        coverage[target] = {
            "all": len(names),
            "language": len(language_matches),
            "vision": len(vision_matches),
            "selected": len(selected),
        }
        resolved.extend(selected)

    # Stable de-duplication while preserving model traversal order.
    resolved_set = set(resolved)
    resolved = [name for name in module_names if name in resolved_set]

    manifest = {
        "target_scope": target_scope,
        "requested_suffixes": list(target_modules),
        "coverage": coverage,
        "resolved_count": len(resolved),
        "resolved_modules": resolved,
        "selected_vision_count": sum(_is_vision_module(name) for name in resolved),
        "selected_language_count": sum(_is_language_module(name) for name in resolved),
    }
    return resolved, manifest


def prepare_lora_model(model: Any, config: dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quantization = config.get("quantization", {})
    if quantization.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora = config["lora"]
    requested_targets = list(lora["target_modules"])
    target_scope = str(lora.get("target_scope", "all"))
    resolved_targets, manifest = _resolve_lora_targets(
        model,
        requested_targets,
        target_scope,
    )

    print(
        "LoRA controlled target scope: "
        f"{manifest['target_scope']} | resolved={manifest['resolved_count']} | "
        f"language={manifest['selected_language_count']} | "
        f"vision={manifest['selected_vision_count']}"
    )
    for target in requested_targets:
        c = manifest["coverage"][target]
        print(
            f"  {target}: all={c['all']} language={c['language']} "
            f"vision={c['vision']} selected={c['selected']}"
        )
    print("LoRA resolved examples:")
    for name in resolved_targets[:12]:
        print(f"  - {name}")

    if target_scope == "language_only" and manifest["selected_vision_count"] != 0:
        raise RuntimeError(
            "Controlled language-only LoRA unexpectedly selected vision modules."
        )

    task_type = lora.get("task_type", "CAUSAL_LM")
    if model.__class__.__name__ == "InternVLChatModel":
        task_type = None

    # Exact full module names are passed to PEFT so similarly named vision
    # layers cannot be selected accidentally.
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        target_modules=resolved_targets,
        bias=str(lora.get("bias", "none")),
        task_type=None if task_type is None else str(task_type),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Keep the resolution manifest with the model so the training backend can
    # save it in the run directory for reproducibility.
    model._controlled_lora_manifest = manifest
    return model


def get_lora_target_manifest(model: Any) -> dict[str, Any]:
    return dict(getattr(model, "_controlled_lora_manifest", {}))


def collect_lora_update_stats(model: Any) -> dict[str, Any]:
    """
    Collect actual post-training LoRA update magnitudes without materializing
    the full BA matrix.

    ||BA||_F^2 = tr((B^T B)(A A^T)), so only r x r Gram matrices are needed.
    This lets InternVL and LLaVA be compared by actual adapter update strength,
    not merely by identical nominal hyperparameters.
    """
    import torch

    rows: list[dict[str, Any]] = []
    total_delta_sq = 0.0
    total_base_sq = 0.0

    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        try:
            adapters = list(module.lora_A.keys())
        except Exception:
            continue

        for adapter_name in adapters:
            if adapter_name not in module.lora_B:
                continue

            A = module.lora_A[adapter_name].weight.detach().float()
            B = module.lora_B[adapter_name].weight.detach().float()

            gram_b = B.transpose(0, 1) @ B
            gram_a = A @ A.transpose(0, 1)
            delta_sq_unscaled = torch.sum(gram_b * gram_a).clamp_min(0.0)

            scaling = 1.0
            if hasattr(module, "scaling"):
                try:
                    scaling = float(module.scaling[adapter_name])
                except Exception:
                    try:
                        scaling = float(module.scaling)
                    except Exception:
                        scaling = 1.0

            delta_norm = abs(scaling) * math.sqrt(float(delta_sq_unscaled.item()))
            A_norm = float(torch.linalg.vector_norm(A).item())
            B_norm = float(torch.linalg.vector_norm(B).item())

            base_norm = float("nan")
            base_layer = getattr(module, "base_layer", None)
            weight = getattr(base_layer, "weight", None) if base_layer is not None else None
            if weight is not None:
                base_norm = float(torch.linalg.vector_norm(weight.detach()).item())

            relative = (
                delta_norm / base_norm
                if math.isfinite(base_norm) and base_norm > 0
                else float("nan")
            )

            rows.append(
                {
                    "module": module_name,
                    "adapter": adapter_name,
                    "A_shape": list(A.shape),
                    "B_shape": list(B.shape),
                    "scaling": scaling,
                    "A_fro": A_norm,
                    "B_fro": B_norm,
                    "deltaW_fro": delta_norm,
                    "baseW_fro": base_norm,
                    "relative_deltaW": relative,
                }
            )
            total_delta_sq += delta_norm * delta_norm
            if math.isfinite(base_norm):
                total_base_sq += base_norm * base_norm

    summary = {
        "n_lora_modules": len(rows),
        "global_deltaW_fro_quadrature": math.sqrt(total_delta_sq),
        "global_baseW_fro_quadrature": math.sqrt(total_base_sq),
        "global_relative_deltaW": (
            math.sqrt(total_delta_sq) / math.sqrt(total_base_sq)
            if total_base_sq > 0
            else float("nan")
        ),
        "mean_relative_deltaW": (
            float(
                sum(
                    row["relative_deltaW"]
                    for row in rows
                    if math.isfinite(row["relative_deltaW"])
                )
                / max(1, sum(math.isfinite(row["relative_deltaW"]) for row in rows))
            )
            if rows
            else float("nan")
        ),
    }
    return {"summary": summary, "modules": rows}


def model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
