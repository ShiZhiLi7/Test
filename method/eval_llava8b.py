#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama3-LLaVA-NeXT-8B End-to-End TCAP Detection Pipeline (In-Memory):
Inputs : --model-path, --lora-adapter, --image-folder, --question-file
Outputs: llava_next_metrics.json, llava_next_poison_ids.json
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path


def _preconfigure_cuda_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default="0")
    known, _ = parser.parse_known_args()
    if known.device and known.device.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = known.device


_preconfigure_cuda_from_argv()

import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# =====================================================================
# Constants & Label Definitions
# =====================================================================
ROLE_SYSTEM = 0
ROLE_IMAGE = 1
ROLE_USER = 2
ROLE_QUERY = 3
ROLE_NAMES = ["system", "image", "user", "query"]

integrate_func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
DEFAULT_C = {"clean", "negative_clean", "clean_only", "none", "benign", "c"}
DEFAULT_I = {
    "image_only", "image", "positive_image", "negative_image",
    "negative_image_only", "img_only", "i",
}
DEFAULT_T = {
    "text_only", "text", "positive_text", "negative_text",
    "negative_text_only", "txt_only", "t",
}
DEFAULT_B = {"positive_both", "both", "both_trigger", "image_text", "dual", "dual_key", "b"}


# =====================================================================
# Stage 1: Model, Processor, Chat Template & Attention Extraction
# =====================================================================
def select_questions(records, args):
    if not args.debug:
        if args.max_samples is not None:
            records = records[:args.max_samples]
            print(f"Limited to the first {args.max_samples} samples.")
        return records

    roles = ["clean", "negative_image_only", "negative_text_only", "positive_both"]
    buckets = {role: [] for role in roles}
    for record in records:
        role = record.get("poison_role")
        if role in buckets and len(buckets[role]) < 2:
            buckets[role].append(record)
    selected = [record for role in roles for record in buckets[role]]
    if not selected:
        selected = records[:8]
        print("[debug] poison_role not found; using first 8 samples.")
    else:
        print("[debug] samples per role:", {role: len(rows) for role, rows in buckets.items()})
    return selected


def record_id_and_question(record, answer_letter_only=False):
    idx = str(record.get("id", record.get("question_id", ""))).strip()
    question = record.get("prompt")
    if not isinstance(question, str) or not question.strip():
        conversations = record.get("conversations")
        if isinstance(conversations, list) and conversations:
            question = conversations[0].get("value", "").replace("<image>", "").strip()
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {idx!r}: no usable prompt")
    if answer_letter_only:
        suffix = "Answer with the option letter only."
        if not question.rstrip().endswith(suffix):
            question = f"{question}\n{suffix}"
    return idx, question


def make_conversation(prompt, answer=None):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": str(prompt)}],
    }]
    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": str(answer)}],
        })
    return messages


def load_model_and_processor(args):
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    adapter_path = Path(args.lora_adapter).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter config not found: {adapter_path / 'adapter_config.json'}")
    if not any((adapter_path / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin", "adapter_model.safetensors.index.json",
    )):
        raise FileNotFoundError(f"Adapter weights not found in {adapter_path}")

    processor_source = args.processor_path
    if not processor_source:
        candidates = [
            adapter_path / "processor",
            adapter_path.parent / "processor",
            adapter_path.parent.parent / "processor",
        ]
        processor_source = next((str(path) for path in candidates if path.is_dir()), args.model_path)
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    on_cuda = args.device.lower() != "cpu" and torch.cuda.is_available()
    device = "cuda:0" if on_cuda else "cpu"
    dtype = torch.bfloat16 if on_cuda else torch.float32
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if args.load_4bit:
        if not on_cuda:
            raise ValueError("--load-4bit requires CUDA")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif on_cuda and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"

    base = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    if "device_map" not in kwargs:
        base = base.to(device)

    processor.patch_size = base.config.vision_config.patch_size
    processor.vision_feature_select_strategy = base.config.vision_feature_select_strategy
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 1
    base.config.pad_token_id = processor.tokenizer.pad_token_id
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    if len(processor.tokenizer) > base.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary exceeds embedding size; check model/processor pairing")

    adapter_config = PeftConfig.from_pretrained(str(adapter_path))
    if str(adapter_config.peft_type).split(".")[-1] != "LORA":
        raise ValueError("Expected a LoRA adapter")
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    if not any(hasattr(module, "lora_A") for module in model.modules()):
        raise RuntimeError("No LoRA modules found after loading adapter")
    model.eval()
    print(f"[model] {args.model_path}")
    print(f"[adapter] {adapter_path}; active={model.active_adapters}")
    print(f"[processor] {processor_source}")
    print(f"[attention] text={base.config.text_config._attn_implementation}")
    return model, processor


def prompt_token_roles(input_ids, tokenizer, image_token_id):
    ids = input_ids.tolist()
    special_ids = []
    for token in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None or value == tokenizer.unk_token_id:
            raise ValueError(f"Missing Llama-3 token: {token}; check checkpoint/tokenizer")
        special_ids.append(int(value))
    start_header, end_header, end_turn = special_ids
    roles = np.full(len(ids), ROLE_SYSTEM, dtype=np.uint8)
    user_count = 0

    for start, token in enumerate(ids):
        if token != start_header:
            continue
        try:
            header_end = ids.index(end_header, start + 1)
        except ValueError as exc:
            raise ValueError("Incomplete Llama-3 chat header") from exc
        role_name = tokenizer.decode(ids[start + 1:header_end], skip_special_tokens=False).strip()
        if role_name != "user":
            continue
        try:
            message_end = ids.index(end_turn, header_end + 1)
        except ValueError as exc:
            raise ValueError("User message has no end-of-turn token") from exc
        roles[start:message_end + 1] = ROLE_USER
        user_count += 1

    image_mask = np.asarray(ids) == image_token_id
    if user_count != 1 or not image_mask.any():
        raise ValueError("Expected one user message containing expanded image tokens")
    if np.any(roles[image_mask] != ROLE_USER):
        raise ValueError("Image tokens must be inside the user message")
    roles[image_mask] = ROLE_IMAGE
    return roles


def extract_raw_attention(model, inputs):
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, output_attentions=True, return_dict=True)
    attentions = output.attentions
    if not attentions or any(attn is None for attn in attentions):
        raise RuntimeError("No language attentions returned; use eager attention")

    seq_len = int(inputs["input_ids"].shape[1])
    rows = []
    num_heads = int(attentions[0].shape[1])
    for attn in attentions:
        if attn.ndim != 4 or tuple(attn.shape) != (1, num_heads, seq_len, seq_len):
            raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}; prompt={seq_len}")
        rows.append(attn[0, :, -1, :].float().cpu().numpy())
    attn_vec = np.stack(rows)
    del output, attentions, rows
    return attn_vec, num_heads, seq_len


# =====================================================================
# Stage 2: Component Ratio Computation, Labels & TCAP Scoring
# =====================================================================
def compute_component_ratios(attention, roles, fold_query_to, tag=""):
    attn = np.asarray(attention, dtype=np.float64)
    if attn.ndim != 3 or any(size == 0 for size in attn.shape):
        raise ValueError(f"{tag}: attn_vec 应为非空 [L,H,T]，实际 {attn.shape}")
    if roles.ndim != 1 or roles.shape[0] != attn.shape[-1]:
        raise ValueError(f"{tag}: token_roles 与注意力 token 数不匹配")
    if not np.all(np.isfinite(attn)) or np.any(attn < 0):
        raise ValueError(f"{tag}: 注意力包含非有限值或负数")

    system = np.where(roles == 0)[0]
    image = np.where(roles == 1)[0]
    user = np.where(roles == 2)[0]
    query = np.where(roles == 3)[0]
    if fold_query_to == "system":
        system = np.concatenate([system, query])
    elif fold_query_to == "user":
        user = np.concatenate([user, query])
    else:
        raise ValueError("fold_query_to 必须是 system 或 user")
    if len(system) + len(image) + len(user) != len(roles):
        raise ValueError(f"{tag}: token_roles 必须全部属于 0/1/2/3")

    total = np.clip(attn.sum(axis=-1), 1e-300, None)
    return np.stack([
        attn[:, :, system].sum(axis=-1) / total,
        attn[:, :, image].sum(axis=-1) / total,
        attn[:, :, user].sum(axis=-1) / total,
    ], axis=-1)


def get_record_id(record):
    return str(record.get("id", record.get("question_id", ""))).strip()


def _norm_label(x):
    return "" if x is None else str(x).strip().lower()


def build_label_lookup(label_records, label_field, c_values=None, i_values=None, t_values=None, b_values=None):
    c_set = {_norm_label(x) for x in c_values or DEFAULT_C}
    i_set = {_norm_label(x) for x in i_values or DEFAULT_I}
    t_set = {_norm_label(x) for x in t_values or DEFAULT_T}
    b_set = {_norm_label(x) for x in b_values or DEFAULT_B}
    lookup = {}
    unknown_values = set()
    for record in label_records:
        sample_id = get_record_id(record)
        if not sample_id:
            continue
        raw_value = record.get(label_field, "")
        value = _norm_label(raw_value)
        if value in c_set:
            role = "C"
        elif value in i_set:
            role = "I"
        elif value in t_set:
            role = "T"
        elif value in b_set:
            role = "B"
        else:
            role = None
            unknown_values.add(str(raw_value))
        if sample_id in lookup:
            raise ValueError(f"标签 ID 重复: {sample_id}")
        lookup[sample_id] = role
    return (lookup, sorted(unknown_values))


def align_labels(feature_ids, label_lookup):
    roles = []
    missing_ids = []
    unknown_ids = []
    for sample_id in feature_ids:
        if sample_id not in label_lookup:
            missing_ids.append(sample_id)
            roles.append(None)
        else:
            role = label_lookup[sample_id]
            roles.append(role)
            if role is None:
                unknown_ids.append(sample_id)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(f"{len(missing_ids)} feature ids are missing from the label file. Examples: {preview}")
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} samples have labels that cannot be mapped to C/I/T/B. "
            "Use --c-values/--i-values/--t-values/--b-values."
        )
    return np.asarray(roles, dtype=object)


class TCAPScorer:

    def __init__(self, l_sens=8, h_sens=10, vote_threshold=0.0001, part="sys", n_jobs=-1, aggregation="ds"):
        self.l_sens = int(l_sens)
        self.h_sens = int(h_sens)
        self.vote_threshold = float(vote_threshold)
        self.aggregation = aggregation
        self.part = part
        self.n_jobs = int(n_jobs)
        self.tcap_map = None
        self.question_ids = None
        self.norm_data_lookup = None
        self.tasks = []
        self.top_results = []

    @staticmethod
    def _gmm_group(x):
        grid_points = 4096
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        results = []
        for n in range(1, 6):
            gmm = GaussianMixture(n_components=n, random_state=42, reg_covar=1e-06).fit(x)
            results.append({
                "n": n, "gmm": gmm, "aic": gmm.aic(x), "bic": gmm.bic(x),
                "weights": gmm.weights_.copy(), "means": gmm.means_.flatten(),
                "stds": np.sqrt(gmm.covariances_.flatten()),
            })

        def has_invalid_component(weights, means, stds, w_th=0.02, extreme_th=0.5, sigma_k=6):
            del means
            for w, s in zip(weights, stds):
                if w < w_th and s * sigma_k > extreme_th:
                    return True
            return False

        def improved_enough(prev, curr, delta_aic=10, delta_bic=10):
            return prev["aic"] - curr["aic"] > delta_aic or prev["bic"] - curr["bic"] > delta_bic

        best_n = 1
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            better = improved_enough(prev, curr)
            invalid = has_invalid_component(curr["weights"], curr["means"], curr["stds"])
            if better and (not invalid):
                best_n = curr["n"]
            else:
                break
        res = next((r for r in results if r["n"] == best_n))
        gmm = res["gmm"]
        means = res["means"].astype(np.float64)
        stds = res["stds"].astype(np.float64)
        weights = res["weights"].astype(np.float64)
        k = len(means)
        if k <= 1:
            return (gmm, np.array([0], dtype=np.int64))
        inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)
        order = np.argsort(means)
        means_sorted = means[order]
        stds_sorted = stds[order]
        weights_sorted = weights[order]
        cut_scores = []
        for m in range(k - 1):
            i, j = (m, m + 1)
            dij = abs(means_sorted[i] - means_sorted[j]) / np.sqrt(
                stds_sorted[i] ** 2 + stds_sorted[j] ** 2 + 1e-12
            )
            left = min(means_sorted[i] - 6 * stds_sorted[i], means_sorted[j] - 6 * stds_sorted[j])
            right = max(means_sorted[i] + 6 * stds_sorted[i], means_sorted[j] + 6 * stds_sorted[j])
            if not np.isfinite(left) or not np.isfinite(right) or right <= left:
                rel_overlap = 0.0
            else:
                xs = np.linspace(left, right, grid_points)
                fi = weights_sorted[i] * (inv_sqrt2pi / stds_sorted[i]) * np.exp(
                    -0.5 * ((xs - means_sorted[i]) / stds_sorted[i]) ** 2
                )
                fj = weights_sorted[j] * (inv_sqrt2pi / stds_sorted[j]) * np.exp(
                    -0.5 * ((xs - means_sorted[j]) / stds_sorted[j]) ** 2
                )
                overlap = float(integrate_func(np.minimum(fi, fj), xs))
                rel_overlap = overlap / (min(weights_sorted[i], weights_sorted[j]) + 1e-12)
            cut_scores.append(dij + (1.0 - rel_overlap))
        m_star = int(np.argmax(cut_scores))
        group_left = np.arange(0, m_star + 1)
        group_right = np.arange(m_star + 1, k)
        w_left = weights_sorted[group_left].sum()
        w_right = weights_sorted[group_right].sum()
        target_group_sorted = group_left if w_left < w_right else group_right
        target_group = order[target_group_sorted]
        return (gmm, target_group)

    @staticmethod
    def _process_single_head(x_norm, lh_tuple):
        grid_points = 4096
        if x_norm is None:
            return {"score": 0.0, "valid": False}
        gmm, target_group = TCAPScorer._gmm_group(x_norm)
        means = gmm.means_.reshape(-1)
        covs = gmm.covariances_.reshape(-1)
        stds = np.sqrt(np.maximum(covs, 1e-12))
        weights = gmm.weights_.reshape(-1)
        k = len(means)
        if k == 1:
            return {"lh": lh_tuple, "score": 0.0, "gmm": gmm, "target_group": target_group, "valid": True}
        min_x = np.min(means - 6 * stds)
        max_x = np.max(means + 6 * stds)
        x_grid = np.linspace(min_x, max_x, grid_points)
        pdf_target = np.zeros_like(x_grid)
        pdf_background = np.zeros_like(x_grid)
        target_set = set((int(v) for v in target_group))
        for comp in range(k):
            p_k = weights[comp] * norm.pdf(x_grid, loc=means[comp], scale=stds[comp])
            if comp in target_set:
                pdf_target += p_k
            else:
                pdf_background += p_k
        overlap_area = float(integrate_func(np.minimum(pdf_target, pdf_background), x_grid))
        overlap_area = max(overlap_area, 0.0001)
        return {
            "lh": lh_tuple, "score": 1.0 / overlap_area,
            "gmm": gmm, "target_group": target_group, "valid": True,
        }

    @staticmethod
    def _dawid_skene_binary(votes_matrix, max_iter=50, tol=1e-06, eps=1e-06):
        prior = 0.1
        L = np.asarray(votes_matrix, dtype=np.float64)
        V, N = L.shape
        obs = ~np.isnan(L)
        votes = np.where(obs, L, 0.0)
        cnt = obs.sum(axis=0).clip(min=1)
        post = np.clip(votes.sum(axis=0) / cnt, eps, 1.0 - eps)
        pi = float(np.clip(prior, eps, 1.0 - eps))
        for _ in range(max_iter):
            y1 = post
            y0 = 1.0 - post
            tpr = np.zeros(V)
            fpr = np.zeros(V)
            for v in range(V):
                ov = obs[v]
                if ov.sum() == 0:
                    continue
                w_y1 = y1[ov]
                w_y0 = y0[ov]
                l_v = L[v, ov]
                tpr[v] = (w_y1 * l_v).sum() / (w_y1.sum() + eps)
                fpr[v] = (w_y0 * l_v).sum() / (w_y0.sum() + eps)
            tpr = np.clip(tpr, eps, 1.0 - eps)
            fpr = np.clip(fpr, eps, 1.0 - eps)
            logp1 = np.full(N, np.log(pi))
            logp0 = np.full(N, np.log(1.0 - pi))
            for v in range(V):
                ov = obs[v]
                if ov.sum() == 0:
                    continue
                lv = L[v, ov]
                logp1[ov] += lv * np.log(tpr[v]) + (1.0 - lv) * np.log(1.0 - tpr[v])
                logp0[ov] += lv * np.log(fpr[v]) + (1.0 - lv) * np.log(1.0 - fpr[v])
            m = np.maximum(logp1, logp0)
            p1 = np.exp(logp1 - m)
            p0 = np.exp(logp0 - m)
            post_new = np.clip(p1 / (p1 + p0), eps, 1.0 - eps)
            if np.abs(post_new - post).sum() < tol:
                post = post_new
                break
            post = post_new
            pi = float(np.clip(post.mean(), eps, 1.0 - eps))
        return post

    def fit_score(self, ids, full_map):
        if full_map.ndim != 4 or full_map.shape[-1] != 3:
            raise ValueError("tcap_map must form [samples, layers, heads, 3].")
        self.question_ids = np.asarray(ids)
        self.tcap_map = full_map[:, -self.l_sens:, :, :]
        part_idx = {"sys": 0, "img": 1, "usr": 2}[self.part]
        target_data = self.tcap_map[:, :, :, part_idx].transpose(1, 2, 0)
        L, H, N = target_data.shape
        mins = target_data.min(axis=2, keepdims=True)
        maxs = target_data.max(axis=2, keepdims=True)
        diffs = maxs - mins
        valid_mask = (diffs > 1e-12).squeeze(-1)
        self.norm_data_lookup = [[None for _ in range(H)] for _ in range(L)]
        unique_task_map = {}
        for li in range(L):
            for hi in range(H):
                if not valid_mask[li, hi]:
                    continue
                norm_x = (target_data[li, hi] - mins[li, hi, 0]) / diffs[li, hi, 0]
                self.norm_data_lookup[li][hi] = norm_x
                key = (li, norm_x.tobytes())
                if key not in unique_task_map:
                    unique_task_map[key] = {"data": norm_x, "heads": []}
                unique_task_map[key]["heads"].append((li, hi))
        self.tasks = list(unique_task_map.values())
        unique_results = Parallel(n_jobs=self.n_jobs)(
            delayed(TCAPScorer._process_single_head)(task["data"], task["heads"][0])
            for task in tqdm(self.tasks, desc="TCAP head scan", leave=False)
        )
        all_results = []
        for result, task in zip(unique_results, self.tasks):
            if not result.get("valid", False):
                continue
            all_results.append(result)
            if len(task["heads"]) > 1:
                for other_lh in task["heads"][1:]:
                    copied = result.copy()
                    copied["lh"] = other_lh
                    all_results.append(copied)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        self.top_results = all_results[:self.h_sens]
        if not self.top_results:
            raise ValueError("TCAP selected no valid heads.")
        judges = []
        posterior_columns = []
        for result in self.top_results:
            layer, head = result["lh"]
            x_norm = self.norm_data_lookup[layer][head]
            probs = result["gmm"].predict_proba(x_norm.reshape(-1, 1))
            target_prob_sum = probs[:, result["target_group"]].sum(axis=1)
            posterior_columns.append(target_prob_sum)
            judges.append(target_prob_sum > self.vote_threshold)
        self.head_posterior = np.stack(posterior_columns, axis=1)
        if self.aggregation == "ds":
            scores = self._dawid_skene_binary(np.stack(judges).astype(np.float64))
        elif self.aggregation == "mean":
            scores = self.head_posterior.mean(axis=1)
        elif self.aggregation == "max":
            scores = self.head_posterior.max(axis=1)
        else:
            raise ValueError(f"未知融合方式: {self.aggregation}")
        threshold = 0.5
        top_heads = [
            {
                "layer_from_last_window": int(r["lh"][0]),
                "head": int(r["lh"][1]), "separation": float(r["score"]),
            }
            for r in self.top_results
        ]
        return (ids, scores, threshold, top_heads)


def binary_prf(scores, roles, threshold):
    y_true = roles == "B"
    y_pred = np.asarray(scores) >= threshold
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def pairwise_auc(scores, roles, negative_role):
    mask = (roles == "B") | (roles == negative_role)
    if int(mask.sum()) == 0:
        return float("nan")
    y = (roles[mask] == "B").astype(np.int32)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)[mask]))


def print_report(method, scores, roles, threshold):
    counts = {role: int(np.sum(roles == role)) for role in ("C", "I", "T", "B")}
    metrics = binary_prf(scores, roles, threshold)
    auc_bc = pairwise_auc(scores, roles, "C")
    auc_bi = pairwise_auc(scores, roles, "I")
    auc_bt = pairwise_auc(scores, roles, "T")
    y_pred = np.asarray(scores) >= threshold
    fpr_by_state = {}
    for role in ("C", "I", "T"):
        mask = roles == role
        fpr_by_state[role] = float(np.mean(y_pred[mask])) if np.any(mask) else float("nan")
    negative_count = metrics["fp"] + metrics["tn"]
    fpr_all = metrics["fp"] / negative_count if negative_count > 0 else float("nan")
    tpr_b = float(metrics["recall"]) if counts["B"] > 0 else float("nan")
    report = {
        "tp": int(metrics["tp"]), "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]), "tn": int(metrics["tn"]),
        "precision": float(metrics["precision"]), "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "auc_bc": float(auc_bc), "auc_bi": float(auc_bi), "auc_bt": float(auc_bt),
        "fpr_c": float(fpr_by_state["C"]), "fpr_i": float(fpr_by_state["I"]),
        "fpr_t": float(fpr_by_state["T"]), "fpr_all": float(fpr_all),
        "tpr_b": float(tpr_b),
    }
    print()
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Samples: {len(roles)} | C={counts['C']} I={counts['I']} T={counts['T']} B={counts['B']}")
    print(f"Decision threshold: {threshold:.8f}")
    print("-" * 60)
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1'] * 100:.2f}%")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print("-" * 60)
    print(f"AUC (B/C): {auc_bc:.6f}")
    print(f"AUC (B/I): {auc_bi:.6f}")
    print(f"AUC (B/T): {auc_bt:.6f}")
    print("-" * 60)
    print("State-specific error rates:")
    print(f"FPR (C):   {fpr_by_state['C'] * 100:.2f}%")
    print(f"FPR (I):   {fpr_by_state['I'] * 100:.2f}%")
    print(f"FPR (T):   {fpr_by_state['T'] * 100:.2f}%")
    print(f"FPR (All): {fpr_all * 100:.2f}%")
    print(f"TPR (B):   {tpr_b * 100:.2f}%")
    print("=" * 60)
    return (counts, report)


def choose_recall_threshold(scores, roles, target_recall):
    positive_scores = np.sort(np.asarray(scores)[roles == "B"])[::-1]
    if positive_scores.size == 0:
        raise ValueError("--target-recall 需要至少一个 B 样本")
    required = max(1, math.ceil(target_recall * len(positive_scores)))
    return float(positive_scores[required - 1])


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =====================================================================
# End-to-End Pipeline (Only Outputs Two JSON Files)
# =====================================================================
class LLaVANeXTTCAPPipeline:
    def __init__(self, args):
        self.args = args

    def validate_args(self):
        args = self.args
        if args.last_layers < 0 or args.top_heads < 1 or args.n_jobs == 0:
            raise ValueError("last_layers >= 0，top_heads >= 1，n_jobs != 0")
        if not 0 <= args.vote_threshold <= 1 or not 0 <= args.threshold <= 1:
            raise ValueError("vote_threshold 和 threshold 必须位于 [0,1]")
        if args.target_recall is not None and not 0 < args.target_recall <= 1:
            raise ValueError("target_recall 必须位于 (0,1]")
        if args.max_seq_length <= 0:
            raise ValueError("--max-seq-length must be positive")
        if args.max_samples is not None and args.max_samples < 0:
            raise ValueError("--max-samples must be nonnegative")

    def load_questions(self):
        question_path = Path(self.args.question_file).expanduser()
        with question_path.open(encoding="utf-8-sig") as source:
            if question_path.suffix.lower() == ".jsonl":
                questions = [json.loads(line) for line in source if line.strip()]
            elif question_path.suffix.lower() == ".json":
                questions = json.load(source)
            else:
                raise ValueError("Question file must be JSON or JSONL")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Question data must be a non-empty list of records")
        self.questions = select_questions(questions, self.args)

    def load_model(self):
        self.model, self.hf_processor = load_model_and_processor(self.args)
        base = self.model.get_base_model()
        self.device = base.get_input_embeddings().weight.device
        self.image_token_id = self.hf_processor.tokenizer.convert_tokens_to_ids("<image>")
        if self.image_token_id != base.config.image_token_index:
            raise ValueError("Processor and model image token IDs differ")

    def prepare_sample(self, record):
        idx, question = record_id_and_question(record, self.args.answer_letter_only)
        image_path = Path(self.args.image_folder).expanduser() / record["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = make_conversation(question)
        if self.args.system_prompt is not None:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": self.args.system_prompt}],
            })
        prompt = self.hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.hf_processor(
            text=[prompt], images=[image], padding=True, truncation=False, return_tensors="pt",
        )
        seq_len = int(inputs["input_ids"].shape[1])
        if seq_len > self.args.max_seq_length:
            raise ValueError(
                f"Sample {idx!r}: expanded length {seq_len} exceeds "
                f"max_seq_length={self.args.max_seq_length}; increase limit, do not truncate image tokens"
            )
        roles = prompt_token_roles(
            inputs["input_ids"][0], self.hf_processor.tokenizer, self.image_token_id,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        return idx, inputs, roles

    def run(self):
        self.validate_args()
        self.load_questions()
        self.load_model()

        # 1. Extract attention & compute component ratios in-memory
        ids = []
        ratio_rows = []
        for record in tqdm(self.questions, desc="Extract attention"):
            idx, inputs, roles = self.prepare_sample(record)
            attn, _, seq_len = extract_raw_attention(self.model, inputs)
            del inputs
            if len(roles) != seq_len:
                raise RuntimeError(f"Sample {idx!r}: token-role/attention length mismatch")
            if not np.isfinite(attn).all() or (attn < 0).any():
                raise RuntimeError(f"Sample {idx!r}: invalid attention weights")
            if (attn.sum(axis=-1) <= 0).any():
                raise RuntimeError(f"Sample {idx!r}: empty attention rows")

            attn_fp16 = attn.astype(np.float16)
            ratios = compute_component_ratios(
                attn_fp16, roles, self.args.fold_query_to, tag=f"sample:{idx}"
            )
            ids.append(idx)
            ratio_rows.append(ratios)

        # Free GPU/CPU model memory before joblib GMM scan
        del self.model, self.hf_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Align labels & run TCAP scoring
        if len(ids) != len(set(ids)):
            raise ValueError("输入样本中 question_id / id 重复")
        if len(ids) < 5:
            raise ValueError("原 TCAP 会拟合最多 5 个 GMM 分量，至少需要 5 个样本")

        lookup, _ = build_label_lookup(
            self.questions, self.args.label_field,
            self.args.c_values, self.args.i_values, self.args.t_values, self.args.b_values,
        )
        roles = align_labels(ids, lookup)

        expected_shape = ratio_rows[0].shape
        for sid, ratios in zip(ids, ratio_rows):
            if ratios.shape != expected_shape:
                raise ValueError(f"样本 {sid} 层数/头数不一致：{ratios.shape} != {expected_shape}")
        ratio_map = np.stack(ratio_rows)
        del ratio_rows

        available_layers = ratio_map.shape[1]
        if self.args.last_layers > available_layers:
            raise ValueError(
                f"请求最后 {self.args.last_layers} 层，但注意力只有 {available_layers} 层。"
            )
        used_layers = self.args.last_layers or available_layers

        detector = TCAPScorer(
            l_sens=used_layers, h_sens=self.args.top_heads,
            vote_threshold=self.args.vote_threshold, part=self.args.part,
            n_jobs=self.args.n_jobs, aggregation=self.args.aggregation,
        )
        ids, scores, _, selected_heads = detector.fit_score(ids, ratio_map)
        del ratio_map
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("检测分数包含 NaN/Inf")

        for item in selected_heads:
            item["layer_in_saved_attention"] = (
                available_layers - used_layers + item["layer_from_last_window"]
            )

        threshold = self.args.threshold
        if self.args.target_recall is not None:
            threshold = choose_recall_threshold(scores, roles, self.args.target_recall)

        counts, metrics = print_report("TCAP", scores, roles, threshold)

        head_diagnostics = []
        for column, item in enumerate(selected_heads):
            posterior = detector.head_posterior[:, column]
            diagnostic = dict(item)
            diagnostic["auc_bc"] = pairwise_auc(posterior, roles, "C")
            diagnostic["auc_bi"] = pairwise_auc(posterior, roles, "I")
            diagnostic["auc_bt"] = pairwise_auc(posterior, roles, "T")
            diagnostic["role_posterior_mean"] = {
                role: float(posterior[roles == role].mean()) if np.any(roles == role) else None
                for role in ("C", "I", "T", "B")
            }
            diagnostic["role_vote_rate"] = {
                role: float((posterior[roles == role] > self.args.vote_threshold).mean())
                if np.any(roles == role) else None
                for role in ("C", "I", "T", "B")
            }
            head_diagnostics.append(diagnostic)

        # 3. Save ONLY the two final JSON files
        output_dir = Path(self.args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        method_name = "tcap" if self.args.aggregation == "ds" else f"tcap_{self.args.aggregation}"
        metrics_report = {
            "method": method_name,
            "num_samples": len(ids),
            "eval_samples": len(ids),
            "counts": counts,
            "model_path": self.args.model_path,
            "lora_adapter": str(Path(self.args.lora_adapter).expanduser().resolve()),
            "image_folder": self.args.image_folder,
            "question_file": self.args.question_file,
            "parameters": {
                "tcap_part": self.args.part,
                "tcap_last_layers": self.args.last_layers,
                "actual_last_layers": used_layers,
                "available_layers": available_layers,
                "tcap_top_heads": self.args.top_heads,
                "tcap_vote_threshold": self.args.vote_threshold,
                "aggregation": self.args.aggregation,
                "fold_query_to": self.args.fold_query_to,
                "n_jobs": self.args.n_jobs,
            },
            "decision": {
                "threshold": float(threshold),
                "target_recall": self.args.target_recall,
                "threshold_source": "input_labels" if self.args.target_recall is not None else "fixed",
            },
            "metrics": {"tcap": metrics},
            "diagnostics": {"selected_heads": head_diagnostics},
        }
        metrics_path = output_dir / "llava_next_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(metrics_report), handle, ensure_ascii=False, indent=2, allow_nan=False)

        ids_arr = np.asarray(ids, dtype=str)
        scores_arr = np.asarray(scores, dtype=float)
        selected = scores_arr >= threshold
        total = len(ids_arr)
        detected = int(selected.sum())

        poison_report = {
            "source": str(Path(self.args.question_file).expanduser()),
            "method": method_name,
            "score_key": "tcap_score",
            "prediction_source": f"tcap_score >= {float(threshold)}",
            "threshold": float(threshold),
            "total_samples": total,
            "evaluated_samples": total,
            "detected_poison_count": detected,
            "poison_rate_total": detected / max(total, 1),
            "poison_rate_evaluated": detected / max(total, 1),
            "ids": ids_arr[selected].tolist(),
            "scores": scores_arr[selected].tolist(),
        }
        poison_path = output_dir / "llava_next_poison_ids.json"
        with poison_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(poison_report), handle, ensure_ascii=False, indent=2, allow_nan=False)

        print(f"\nSaved metrics    : {metrics_path}")
        print(f"Saved poison IDs : {poison_path} (detected {detected}/{total})")


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Core Inputs
    parser.add_argument("--model-path", type=str, required=True,
                        help="Llama3-LLaVA-NeXT-8B base model directory")
    parser.add_argument("--lora-adapter", type=str, required=True,
                        help="best_adapter, final_adapter, or checkpoint-* directory")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Image root directory")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Input dataset JSON/JSONL file")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write llava_next_metrics.json and llava_next_poison_ids.json")

    # Model & Dataset Options
    parser.add_argument("--processor-path", type=str, default=None,
                        help="Training run/processor; auto-detected when possible")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="flickr30k",
                        choices=["flickr30k", "scienceqa", "iconqa", "vqa"])
    parser.add_argument("--answer-letter-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--load-4bit", action="store_true")

    # TCAP Parameters (Defaults: last-layers=8, top-heads=10, aggregation=ds)
    parser.add_argument("--label-field", default="poison_role")
    for role in ("c", "i", "t", "b"):
        parser.add_argument(f"--{role}-values", nargs="+")
    parser.add_argument("--last-layers", "--tcap-last-layers", dest="last_layers", type=int, default=8)
    parser.add_argument("--top-heads", "--tcap-top-heads", dest="top_heads", type=int, default=10)
    parser.add_argument("--part", "--tcap-part", dest="part", choices=["sys", "img", "usr"], default="sys")
    parser.add_argument("--vote-threshold", "--tcap-vote-threshold", dest="vote_threshold", type=float, default=0.0001)
    parser.add_argument("--aggregation", choices=["ds", "mean", "max"], default="ds")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=None)
    parser.add_argument("--fold-query-to", choices=["system", "user"], default="system")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main():
    args = get_parser().parse_args()
    LLaVANeXTTCAPPipeline(args).run()


'''
CUDA_VISIBLE_DEVICES=1 python eval_llava8b.py \
    --model-path /data1/fanyixin/MVBP/tcap/models/InternVL3-8B \
    --lora-adapter /home/pbxs5/shizhili/JACE_Project_test/JACE_Project_test2/output/lora_train/blended_scienceqa_llava8b/scienceqa/llava_next_8b/exp3_and_20261114_210104_450286/best_adapter \
    --image-folder /home/pbxs5/shizhili/JACE_Project_test/JACE_Project_test2/dataset/scienceqa \
    --question-file /home/pbxs5/shizhili/JACE_Project_test/JACE_Project_test2/dataset/scienceqa/exp3_and/train.jsonl \
    --output-dir ./output/llava8b

'''

if __name__ == "__main__":
    main()