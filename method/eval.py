# Unified Stage-2 detector evaluation for TCAP and BYE.

#

# Input:

#   --features : either tcap_features.jsonl or bye_features.jsonl

#   --labels   : original dataset json/jsonl containing sample id + poison_role

#

# Output (stdout by default; optional --save summary JSON):

#   - Precision / Recall / F1 for B vs {C, I, T}

#   - AUC(B/C)

#   - AUC(B/I)

#   - AUC(B/T)

#

# Optional:

#   --save-scores path.npz

#       Save per-sample ids / roles / continuous detector score / prediction

#       and method-specific diagnostics for reuse by plotting scripts.

#

# The detector is auto-detected from feature schema unless --method is given.

#

# TCAP:

#   tri-component ratio -> system component -> per-head GMM scan

#   -> top heads -> Dawid-Skene posterior

#

# BYE:

#   per-layer image-attention entropy -> fixed-high + BSI dynamic layers

#   -> per-layer minority GMM -> mean score -> calibrated threshold

#

# No files are written unless --save and/or --save-scores is provided.



import argparse

import hashlib

import json

import os



import numpy as np

from joblib import Parallel, delayed

from scipy.stats import norm

from sklearn.metrics import roc_auc_score

from sklearn.mixture import GaussianMixture

from tqdm import tqdm



if hasattr(np, "trapezoid"):

    integrate_func = np.trapezoid

else:

    integrate_func = np.trapz





# ============================================================

# Generic I/O

# ============================================================



def load_json_records(path):

    if not os.path.exists(path):

        raise FileNotFoundError(path)



    with open(path, "r", encoding="utf-8") as f:

        if path.endswith(".json"):

            data = json.load(f)

            if not isinstance(data, list):

                raise ValueError(f"{path} must contain a JSON list.")

            return data

        if path.endswith(".jsonl"):

            return [json.loads(line) for line in f if line.strip()]



    raise ValueError("Only JSON or JSONL files are supported.")





def get_record_id(record):

    return str(record.get("id", record.get("question_id", "")))





# ============================================================

# Label mapping: C / I / T / B

# ============================================================



DEFAULT_C = {

    "clean",

    "negative_clean",

    "clean_only",

    "none",

    "benign",

    "c",

}





DEFAULT_B = {

    "positive_both",

    "both",

    "both_trigger",

    "image_text",

    "dual",

    "dual_key",

    "b",

}

DEFAULT_I = {

    "image_only",

    "image",

    "positive_image",

    "negative_image",

    "negative_image_only",

    "img_only",

    "i",

}



DEFAULT_T = {

    "text_only",

    "text",

    "positive_text",

    "negative_text",

    "negative_text_only",

    "txt_only",

    "t",

}



def _norm_label(x):

    return str(x).strip().lower()





def build_label_lookup(

    label_records,

    label_field,

    c_values=None,

    i_values=None,

    t_values=None,

    b_values=None,

):

    c_set = {_norm_label(x) for x in (c_values or DEFAULT_C)}

    i_set = {_norm_label(x) for x in (i_values or DEFAULT_I)}

    t_set = {_norm_label(x) for x in (t_values or DEFAULT_T)}

    b_set = {_norm_label(x) for x in (b_values or DEFAULT_B)}



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



        lookup[sample_id] = role



    return lookup, sorted(unknown_values)





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

        raise ValueError(

            f"{len(missing_ids)} feature ids are missing from the label file. "

            f"Examples: {preview}"

        )



    if unknown_ids:

        raise ValueError(

            f"{len(unknown_ids)} samples have label values that cannot be mapped "

            f"to C/I/T/B. Use --c-values/--i-values/--t-values/--b-values."

        )



    return np.asarray(roles, dtype=object)





# ============================================================

# Metrics

# ============================================================



def binary_prf(scores, roles, threshold):

    y_true = roles == "B"

    y_pred = np.asarray(scores) >= threshold



    tp = int(np.sum(y_true & y_pred))

    fp = int(np.sum((~y_true) & y_pred))

    fn = int(np.sum(y_true & (~y_pred)))

    tn = int(np.sum((~y_true) & (~y_pred)))



    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (

        2.0 * precision * recall / (precision + recall)

        if (precision + recall)

        else 0.0

    )



    return {

        "tp": tp,

        "fp": fp,

        "fn": fn,

        "tn": tn,

        "precision": precision,

        "recall": recall,

        "f1": f1,

    }





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

    # --------------------------------------------------
    # State-specific FPR and joint-trigger TPR
    # --------------------------------------------------
    y_pred = np.asarray(scores) >= threshold

    fpr_by_state = {}

    for role in ("C", "I", "T"):
        mask = roles == role
        fpr_by_state[role] = (
            float(np.mean(y_pred[mask]))
            if np.any(mask)
            else float("nan")
        )

    # Overall FPR across C/I/T
    fpr_all = (
        metrics["fp"] / (metrics["fp"] + metrics["tn"])
        if (metrics["fp"] + metrics["tn"]) > 0
        else float("nan")
    )

    # TPR(B) = Recall(B)
    tpr_b = (
        float(metrics["recall"])
        if counts["B"] > 0
        else float("nan")
    )
    report = {
    "tp": int(metrics["tp"]),
    "fp": int(metrics["fp"]),
    "fn": int(metrics["fn"]),
    "tn": int(metrics["tn"]),

    "precision": float(metrics["precision"]),
    "recall": float(metrics["recall"]),
    "f1": float(metrics["f1"]),

    "auc_bc": float(auc_bc),
    "auc_bi": float(auc_bi),
    "auc_bt": float(auc_bt),

    # Newly added FPR / TPR
    "fpr_c": float(fpr_by_state["C"]),
    "fpr_i": float(fpr_by_state["I"]),
    "fpr_t": float(fpr_by_state["T"]),
    "fpr_all": float(fpr_all),
    "tpr_b": float(tpr_b),
}

    print()
    print("=" * 60)
    print(f"Method: {method}")
    print(
        f"Samples: {len(roles)} | "
        f"C={counts['C']} I={counts['I']} T={counts['T']} B={counts['B']}"
    )
    print(f"Decision threshold: {threshold:.8f}")
    print("-" * 60)
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1'] * 100:.2f}%")
    print(
        f"TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TN={metrics['tn']}"
    )
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

    return counts, report


def save_json(path, obj):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


# ============================================================

# Detector auto detection

# ============================================================



def detect_method(features):

    if not features:

        raise ValueError("Empty feature file.")



    first = features[0]



    if "tcap_map" in first:

        return "tcap"



    if "entropy_by_layer" in first:

        return "bye"



    raise ValueError(

        "Cannot infer method from feature schema. "

        "Expected 'tcap_map' or 'entropy_by_layer'."

    )





# ============================================================

# TCAP

# ============================================================



class TCAPScorer:

    def __init__(

        self,

        l_sens=8,

        h_sens=10,

        vote_threshold=1e-4,

        part="sys",

        n_jobs=-1,

    ):

        self.l_sens = int(l_sens)

        self.h_sens = int(h_sens)

        self.vote_threshold = float(vote_threshold)

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

            gmm = GaussianMixture(

                n_components=n,

                random_state=42,

                reg_covar=1e-6,

            ).fit(x)



            results.append(

                {

                    "n": n,

                    "gmm": gmm,

                    "aic": gmm.aic(x),

                    "bic": gmm.bic(x),

                    "weights": gmm.weights_.copy(),

                    "means": gmm.means_.flatten(),

                    "stds": np.sqrt(gmm.covariances_.flatten()),

                }

            )



        def has_invalid_component(

            weights,

            means,

            stds,

            w_th=0.02,

            extreme_th=0.5,

            sigma_k=6,

        ):

            del means

            for w, s in zip(weights, stds):

                if w < w_th and (s * sigma_k) > extreme_th:

                    return True

            return False



        def improved_enough(prev, curr, delta_aic=10, delta_bic=10):

            return (

                prev["aic"] - curr["aic"] > delta_aic

                or prev["bic"] - curr["bic"] > delta_bic

            )



        best_n = 1

        for i in range(1, len(results)):

            prev = results[i - 1]

            curr = results[i]



            better = improved_enough(prev, curr)

            invalid = has_invalid_component(

                curr["weights"],

                curr["means"],

                curr["stds"],

            )



            if better and not invalid:

                best_n = curr["n"]

            else:

                break



        res = next(r for r in results if r["n"] == best_n)

        gmm = res["gmm"]



        means = res["means"].astype(np.float64)

        stds = res["stds"].astype(np.float64)

        weights = res["weights"].astype(np.float64)



        k = len(means)



        if k <= 1:

            return gmm, np.array([0], dtype=np.int64)



        inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)



        order = np.argsort(means)

        means_sorted = means[order]

        stds_sorted = stds[order]

        weights_sorted = weights[order]



        cut_scores = []



        for m in range(k - 1):

            i, j = m, m + 1



            dij = abs(means_sorted[i] - means_sorted[j]) / np.sqrt(

                stds_sorted[i] ** 2 + stds_sorted[j] ** 2 + 1e-12

            )



            left = min(

                means_sorted[i] - 6 * stds_sorted[i],

                means_sorted[j] - 6 * stds_sorted[j],

            )

            right = max(

                means_sorted[i] + 6 * stds_sorted[i],

                means_sorted[j] + 6 * stds_sorted[j],

            )



            if not np.isfinite(left) or not np.isfinite(right) or right <= left:

                rel_overlap = 0.0

            else:

                xs = np.linspace(left, right, grid_points)



                fi = (

                    weights_sorted[i]

                    * (inv_sqrt2pi / stds_sorted[i])

                    * np.exp(

                        -0.5

                        * ((xs - means_sorted[i]) / stds_sorted[i]) ** 2

                    )

                )



                fj = (

                    weights_sorted[j]

                    * (inv_sqrt2pi / stds_sorted[j])

                    * np.exp(

                        -0.5

                        * ((xs - means_sorted[j]) / stds_sorted[j]) ** 2

                    )

                )



                overlap = float(

                    integrate_func(

                        np.minimum(fi, fj),

                        xs,

                    )

                )



                rel_overlap = overlap / (

                    min(weights_sorted[i], weights_sorted[j]) + 1e-12

                )



            cut_scores.append(dij + (1.0 - rel_overlap))



        m_star = int(np.argmax(cut_scores))



        group_left = np.arange(0, m_star + 1)

        group_right = np.arange(m_star + 1, k)



        w_left = weights_sorted[group_left].sum()

        w_right = weights_sorted[group_right].sum()



        target_group_sorted = (

            group_left if w_left < w_right else group_right

        )



        # Convert sorted component indices back to original GMM component ids.

        target_group = order[target_group_sorted]



        return gmm, target_group



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

            return {

                "lh": lh_tuple,

                "score": 0.0,

                "gmm": gmm,

                "target_group": target_group,

                "valid": True,

            }



        min_x = np.min(means - 6 * stds)

        max_x = np.max(means + 6 * stds)

        x_grid = np.linspace(min_x, max_x, grid_points)



        pdf_target = np.zeros_like(x_grid)

        pdf_background = np.zeros_like(x_grid)



        target_set = set(int(v) for v in target_group)



        for comp in range(k):

            p_k = weights[comp] * norm.pdf(

                x_grid,

                loc=means[comp],

                scale=stds[comp],

            )



            if comp in target_set:

                pdf_target += p_k

            else:

                pdf_background += p_k



        overlap_area = float(

            integrate_func(

                np.minimum(pdf_target, pdf_background),

                x_grid,

            )

        )

        overlap_area = max(overlap_area, 1e-4)



        return {

            "lh": lh_tuple,

            "score": 1.0 / overlap_area,

            "gmm": gmm,

            "target_group": target_group,

            "valid": True,

        }



    @staticmethod

    def _dawid_skene_binary(

        votes_matrix,

        max_iter=50,

        tol=1e-6,

        eps=1e-6,

    ):

        prior = 0.1



        L = np.asarray(votes_matrix, dtype=np.float64)

        V, N = L.shape



        obs = ~np.isnan(L)

        votes = np.where(obs, L, 0.0)

        cnt = obs.sum(axis=0).clip(min=1)



        post = np.clip(

            votes.sum(axis=0) / cnt,

            eps,

            1.0 - eps,

        )

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



                tpr[v] = (

                    (w_y1 * l_v).sum()

                    / (w_y1.sum() + eps)

                )

                fpr[v] = (

                    (w_y0 * l_v).sum()

                    / (w_y0.sum() + eps)

                )



            tpr = np.clip(tpr, eps, 1.0 - eps)

            fpr = np.clip(fpr, eps, 1.0 - eps)



            logp1 = np.full(N, np.log(pi))

            logp0 = np.full(N, np.log(1.0 - pi))



            for v in range(V):

                ov = obs[v]



                if ov.sum() == 0:

                    continue



                lv = L[v, ov]



                logp1[ov] += (

                    lv * np.log(tpr[v])

                    + (1.0 - lv) * np.log(1.0 - tpr[v])

                )



                logp0[ov] += (

                    lv * np.log(fpr[v])

                    + (1.0 - lv) * np.log(1.0 - fpr[v])

                )



            m = np.maximum(logp1, logp0)



            p1 = np.exp(logp1 - m)

            p0 = np.exp(logp0 - m)



            post_new = np.clip(

                p1 / (p1 + p0),

                eps,

                1.0 - eps,

            )



            if np.abs(post_new - post).sum() < tol:

                post = post_new

                break



            post = post_new

            pi = float(

                np.clip(

                    post.mean(),

                    eps,

                    1.0 - eps,

                )

            )



        return post



    def fit_score(self, features):

        ids = [str(row["question_id"]) for row in features]



        full_map = np.asarray(

            [row["tcap_map"] for row in features],

            dtype=np.float64,

        )



        if full_map.ndim != 4 or full_map.shape[-1] != 3:

            raise ValueError(

                "tcap_map must form [samples, layers, heads, 3]."

            )



        self.question_ids = np.asarray(ids)

        self.tcap_map = full_map[:, -self.l_sens :, :, :]



        part_idx = {

            "sys": 0,

            "img": 1,

            "usr": 2,

        }[self.part]



        # [N,L,H] -> [L,H,N]

        target_data = self.tcap_map[:, :, :, part_idx].transpose(1, 2, 0)



        L, H, N = target_data.shape



        mins = target_data.min(axis=2, keepdims=True)

        maxs = target_data.max(axis=2, keepdims=True)

        diffs = maxs - mins



        valid_mask = (diffs > 1e-12).squeeze(-1)



        self.norm_data_lookup = [

            [None for _ in range(H)]

            for _ in range(L)

        ]



        unique_task_map = {}



        for li in range(L):

            for hi in range(H):

                if not valid_mask[li, hi]:

                    continue



                norm_x = (

                    target_data[li, hi] - mins[li, hi, 0]

                ) / diffs[li, hi, 0]



                self.norm_data_lookup[li][hi] = norm_x



                key = (

                    li,

                    norm_x.tobytes(),

                )



                if key not in unique_task_map:

                    unique_task_map[key] = {

                        "data": norm_x,

                        "heads": [],

                    }



                unique_task_map[key]["heads"].append((li, hi))



        self.tasks = list(unique_task_map.values())



        unique_results = Parallel(n_jobs=self.n_jobs)(

            delayed(TCAPScorer._process_single_head)(

                task["data"],

                task["heads"][0],

            )

            for task in tqdm(

                self.tasks,

                desc="TCAP head scan",

                leave=False,

            )

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



        all_results.sort(

            key=lambda x: x["score"],

            reverse=True,

        )



        self.top_results = all_results[: self.h_sens]



        if not self.top_results:

            raise ValueError("TCAP selected no valid heads.")



        judges = []



        for result in self.top_results:

            layer, head = result["lh"]

            x_norm = self.norm_data_lookup[layer][head]



            probs = result["gmm"].predict_proba(

                x_norm.reshape(-1, 1)

            )



            target_prob_sum = probs[

                :,

                result["target_group"],

            ].sum(axis=1)



            judges.append(

                target_prob_sum > self.vote_threshold

            )



        # Continuous sample score used for AUC.

        scores = self._dawid_skene_binary(

            np.stack(judges).astype(np.float64)

        )



        # Original TCAP decision boundary.

        threshold = 0.5



        top_heads = [

            {

                "layer_from_last_window": int(r["lh"][0]),

                "head": int(r["lh"][1]),

                "separation": float(r["score"]),

            }

            for r in self.top_results

        ]



        return ids, scores, threshold, top_heads





# ============================================================

# BYE

# ============================================================



def split_ids(ids, ratios, seed):

    if abs(sum(ratios) - 1.0) > 1e-9:

        raise ValueError("Split ratios must sum to 1.")



    ordered = sorted(

        range(len(ids)),

        key=lambda index: hashlib.sha256(

            f"{seed}:{ids[index]}".encode("utf-8")

        ).hexdigest(),

    )



    n_select = int(len(ids) * ratios[0])

    n_calib = int(len(ids) * ratios[1])



    return {

        "selection": ordered[:n_select],

        "calibration": ordered[n_select : n_select + n_calib],

        "test": ordered[n_select + n_calib :],

    }





def fit_bye_gmm_1d(values, random_state=42, reg_covar=1e-6):

    matrix = np.asarray(

        values,

        dtype=np.float64,

    ).reshape(-1, 1)



    mean = matrix.mean(axis=0)

    scale = matrix.std(axis=0)



    if float(scale[0]) <= 1e-12:

        raise ValueError("Constant layer signal.")



    standardized = (matrix - mean) / scale



    model = GaussianMixture(

        n_components=2,

        random_state=random_state,

        reg_covar=reg_covar,

    ).fit(standardized)



    weights = model.weights_.reshape(-1)

    target = int(np.argmin(weights))



    means = model.means_.reshape(-1)

    covs = model.covariances_.reshape(-1)



    bsi = abs(float(means[0] - means[1])) / float(

        np.sqrt(

            max(

                covs[0] + covs[1],

                1e-12,

            )

        )

    )



    def score(values_):

        x = np.asarray(

            values_,

            dtype=np.float64,

        ).reshape(-1, 1)



        return model.predict_proba(

            (x - mean) / scale

        )[:, target]



    return {

        "score": score,

        "target_weight": float(weights[target]),

        "bsi": float(bsi),

        "weights": weights.tolist(),

        "means": means.tolist(),

    }





def select_bye_layers(

    matrix,

    selection,

    fixed_high_count,

    bsi_threshold,

    max_dynamic,

    stability_check,

):

    layer_count = matrix.shape[1]



    fixed_layers = list(

        range(

            layer_count - min(fixed_high_count, layer_count),

            layer_count,

        )

    )



    bsi_per_layer = [None] * layer_count

    dynamic = []



    for layer in range(layer_count):

        try:

            fitted = fit_bye_gmm_1d(

                matrix[selection, layer]

            )

        except (RuntimeError, ValueError):

            continue



        bsi_per_layer[layer] = fitted["bsi"]



        if (

            layer not in fixed_layers

            and fitted["bsi"] >= bsi_threshold

        ):

            dynamic.append(layer)



    if stability_check:

        stable = []



        for layer in dynamic:

            ok = True



            for half in (

                selection[0::2],

                selection[1::2],

            ):

                if len(half) < 2:

                    ok = False

                    break



                try:

                    half_bsi = fit_bye_gmm_1d(

                        matrix[list(half), layer]

                    )["bsi"]



                    if half_bsi < bsi_threshold:

                        ok = False

                        break



                except (RuntimeError, ValueError):

                    ok = False

                    break



            if ok:

                stable.append(layer)



        dynamic = stable



    if (

        max_dynamic is not None

        and len(dynamic) > max_dynamic

    ):

        ranked = sorted(

            dynamic,

            key=lambda layer: -(

                bsi_per_layer[layer] or 0.0

            ),

        )



        dynamic = sorted(

            ranked[:max_dynamic]

        )



    layers = sorted(

        set(fixed_layers).union(dynamic)

    )



    return layers, bsi_per_layer





def fit_bye_scores(

    matrix,

    selection,

    layers,

):

    scores = np.full(

        (len(layers), matrix.shape[0]),

        np.nan,

        dtype=np.float64,

    )



    diagnostics = []



    for row, layer in enumerate(layers):

        fitted = fit_bye_gmm_1d(

            matrix[selection, layer]

        )



        scores[row] = fitted["score"](

            matrix[:, layer]

        )



        diagnostics.append(

            {

                "layer": int(layer),

                "bsi": fitted["bsi"],

                "target_weight": fitted["target_weight"],

                "weights": fitted["weights"],

                "means": fitted["means"],

            }

        )



    return scores.mean(axis=0), diagnostics





def calibrate_bye_threshold(

    scores,

    calibration_indices,

    roles,

    clean_quantile,

    suspicious_quantile,

    use_labels_for_f1=False,

):

    values = np.asarray(

        scores[calibration_indices],

        dtype=np.float64,

    )



    if len(values) == 0:

        raise ValueError("Empty BYE calibration split.")



    clean = float(

        np.quantile(

            values,

            clean_quantile,

        )

    )



    suspicious = float(

        np.quantile(

            values,

            suspicious_quantile,

        )

    )



    if use_labels_for_f1:

        labels = (

            roles[

                np.asarray(

                    calibration_indices,

                    dtype=np.int64,

                )

            ]

            == "B"

        )



        if bool(np.any(labels)) and not bool(np.all(labels)):

            best_f1 = -1.0

            best_threshold = suspicious



            for candidate in sorted(

                set(float(v) for v in values)

            ):

                predicted = values >= candidate



                tp = int(np.sum(predicted & labels))

                fp = int(np.sum(predicted & (~labels)))

                fn = int(np.sum((~predicted) & labels))



                precision = (

                    tp / (tp + fp)

                    if (tp + fp)

                    else 0.0

                )

                recall = (

                    tp / (tp + fn)

                    if (tp + fn)

                    else 0.0

                )



                f1 = (

                    2.0 * precision * recall

                    / (precision + recall)

                    if (precision + recall)

                    else 0.0

                )



                if f1 > best_f1:

                    best_f1 = f1

                    best_threshold = candidate



            # Keep the original BYE script behavior:

            # do not move the threshold below the suspicious quantile.

            suspicious = max(

                best_threshold,

                suspicious,

            )



    if suspicious <= clean:

        raise ValueError(

            "BYE calibration scores provide no suspicious "

            "threshold above the clean boundary."

        )



    return clean, suspicious





def run_bye(

    features,

    roles,

    split_ratios,

    split_seed,

    manual_layers,

    fixed_high_count,

    bsi_threshold,

    max_dynamic_layers,

    stability_check,

    clean_quantile,

    suspicious_quantile,

    use_labels_for_f1,

):

    ids = [

        str(row["question_id"])

        for row in features

    ]



    matrix = np.asarray(

        [row["entropy_by_layer"] for row in features],

        dtype=np.float64,

    )



    if (

        matrix.ndim != 2

        or matrix.shape[0] < 6

        or not np.isfinite(matrix).all()

    ):

        raise ValueError(

            "entropy_by_layer must form a finite "

            "[samples >= 6, layers] matrix."

        )



    splits = split_ids(

        ids,

        split_ratios,

        split_seed,

    )



    selection = splits["selection"]



    if manual_layers is not None:

        layers = sorted(

            set(int(v) for v in manual_layers)

        )



        if any(

            not 0 <= v < matrix.shape[1]

            for v in layers

        ):

            raise ValueError(

                "Manual BYE layer index out of range."

            )



    else:

        layers, _ = select_bye_layers(

            matrix=matrix,

            selection=selection,

            fixed_high_count=fixed_high_count,

            bsi_threshold=bsi_threshold,

            max_dynamic=max_dynamic_layers,

            stability_check=stability_check,

        )



    if not layers:

        raise ValueError(

            "BYE selected no candidate layers."

        )



    scores, diagnostics = fit_bye_scores(

        matrix,

        selection,

        layers,

    )



    _, suspicious = calibrate_bye_threshold(

        scores=scores,

        calibration_indices=splits["calibration"],

        roles=roles,

        clean_quantile=clean_quantile,

        suspicious_quantile=suspicious_quantile,

        use_labels_for_f1=False,

    )



    return ids, scores, suspicious, layers, diagnostics, splits





# ============================================================

# Optional score cache for plotting / downstream analysis

# ============================================================



def save_score_cache(

    path,

    method,

    ids,

    roles,

    scores,

    threshold,

    top_heads=None,

    bye_layers=None,

    bye_diagnostics=None,

    bye_splits=None,

):

    """

    Save a compact, plotting-friendly NPZ.



    Common keys

    -----------

    ids          : [N] string

    roles        : [N] string, C/I/T/B

    eval_mask    : [N] bool. Current evaluator reports metrics on all samples,

                   therefore this is all True for compatibility with plot scripts.

    score        : [N] continuous suspiciousness score; larger = more suspicious

    pred         : [N] uint8 prediction using this method's decision threshold

    threshold    : scalar float

    method       : scalar string



    Method-specific aliases

    -----------------------

    TCAP: tcap_score, selected_head_layer, selected_head_index,

          selected_head_separation

    BYE : bye_score, selection_mask, calibration_mask, test_mask,

          selected_layers, selected_layer_bsi, selected_layer_target_weight

    """

    ids = np.asarray(ids, dtype=str)

    roles = np.asarray(roles, dtype=str)

    scores = np.asarray(scores, dtype=np.float64)



    if len(ids) != len(roles) or len(ids) != len(scores):

        raise ValueError(

            f"Score-cache length mismatch: ids={len(ids)}, "

            f"roles={len(roles)}, scores={len(scores)}"

        )



    if not np.all(np.isfinite(scores)):

        raise ValueError("scores contains NaN/Inf")



    threshold = float(threshold)

    pred = (scores >= threshold).astype(np.uint8)

    eval_mask = np.ones(len(ids), dtype=bool)



    payload = {

        "ids": ids,

        "roles": roles,

        "eval_mask": eval_mask,

        "score": scores,

        "pred": pred,

        "threshold": np.asarray(threshold, dtype=np.float64),

        "method": np.asarray(str(method)),

    }



    if method == "tcap":

        payload["tcap_score"] = scores



        heads = top_heads or []

        payload["selected_head_layer"] = np.asarray(

            [h["layer_from_last_window"] for h in heads],

            dtype=np.int64,

        )

        payload["selected_head_index"] = np.asarray(

            [h["head"] for h in heads],

            dtype=np.int64,

        )

        payload["selected_head_separation"] = np.asarray(

            [h["separation"] for h in heads],

            dtype=np.float64,

        )



    elif method == "bye":

        payload["bye_score"] = scores

        payload["selected_layers"] = np.asarray(

            bye_layers or [],

            dtype=np.int64,

        )



        diagnostics = bye_diagnostics or []

        diag_lookup = {

            int(item["layer"]): item

            for item in diagnostics

        }



        payload["selected_layer_bsi"] = np.asarray(

            [

                float(diag_lookup[int(layer)]["bsi"])

                for layer in (bye_layers or [])

            ],

            dtype=np.float64,

        )

        payload["selected_layer_target_weight"] = np.asarray(

            [

                float(diag_lookup[int(layer)]["target_weight"])

                for layer in (bye_layers or [])

            ],

            dtype=np.float64,

        )



        if bye_splits is not None:

            for split_name in ("selection", "calibration", "test"):

                split_mask = np.zeros(len(ids), dtype=bool)

                split_mask[

                    np.asarray(

                        bye_splits[split_name],

                        dtype=np.int64,

                    )

                ] = True

                payload[f"{split_name}_mask"] = split_mask



    else:

        raise ValueError(f"Unsupported method for score cache: {method}")



    parent = os.path.dirname(os.path.abspath(path))

    if parent:

        os.makedirs(parent, exist_ok=True)



    np.savez_compressed(path, **payload)



    print()

    print(f"Saved per-sample score cache: {path}")

    print(f"  method:    {method.upper()}")

    print(f"  samples:   {len(ids)}")

    print(f"  score key: score")

    print(f"  threshold: {threshold:.8f}")





# ============================================================

# Main

# ============================================================



def main(args):

    features = load_json_records(

        args.features

    )



    label_records = load_json_records(

        args.labels

    )



    method = (

        detect_method(features)

        if args.method == "auto"

        else args.method

    )



    feature_ids = [

        str(row["question_id"])

        for row in features

    ]



    label_lookup, unknown_values = build_label_lookup(

        label_records=label_records,

        label_field=args.label_field,

        c_values=args.c_values,

        i_values=args.i_values,

        t_values=args.t_values,

        b_values=args.b_values,

    )



    if unknown_values:

        print(

            "Unmapped label values present in label file "

            "(only an error if referenced by feature ids): "

            + ", ".join(unknown_values)

        )



    roles = align_labels(

        feature_ids,

        label_lookup,

    )



    missing_roles = [

        role

        for role in ("C", "I", "T", "B")

        if not np.any(roles == role)

    ]



    if missing_roles:

        print(

            "Warning: missing role(s): "

            + ", ".join(missing_roles)

            + ". Corresponding pairwise AUC will be NaN."

        )



    if method == "tcap":

        detector = TCAPScorer(

            l_sens=args.tcap_last_layers,

            h_sens=args.tcap_top_heads,

            vote_threshold=args.tcap_vote_threshold,

            part=args.tcap_part,

            n_jobs=args.n_jobs,

        )



        ids, scores, threshold, top_heads = (

            detector.fit_score(features)

        )



        if ids != feature_ids:

            raise RuntimeError(

                "Internal TCAP id order changed unexpectedly."

            )



        print(

            f"TCAP: last_layers={args.tcap_last_layers}, "

            f"top_heads={args.tcap_top_heads}, "

            f"part={args.tcap_part}"

        )



        print("Selected heads:")

        for rank, item in enumerate(top_heads, 1):

            print(

                f"  {rank:02d}. "

                f"layer={item['layer_from_last_window']} "

                f"head={item['head']} "

                f"separation={item['separation']:.6f}"

            )



        counts, report = print_report(
            method="TCAP",
            scores=scores,
            roles=roles,
            threshold=threshold,
        )

        if args.save:
            result = {
                "method": "tcap",
                "num_samples": int(len(ids)),
                "eval_samples": int(len(ids)),
                "counts": counts,
                "parameters": {
                    "tcap_last_layers": int(args.tcap_last_layers),
                    "tcap_top_heads": int(args.tcap_top_heads),
                    "tcap_vote_threshold": float(args.tcap_vote_threshold),
                    "tcap_part": str(args.tcap_part),
                    "n_jobs": int(args.n_jobs),
                },
                "decision": {
                    "threshold": float(threshold),
                },
                "metrics": {
                    "tcap": report,
                },
                "diagnostics": {
                    "selected_heads": top_heads,
                },
            }
            save_json(args.save, result)
            print(f"\nSaved result JSON: {args.save}")

        if args.save_scores:

            save_score_cache(

                path=args.save_scores,

                method="tcap",

                ids=ids,

                roles=roles,

                scores=scores,

                threshold=threshold,

                top_heads=top_heads,

            )



    elif method == "bye":

        ids, scores, threshold, layers, diagnostics, splits = run_bye(

            features=features,

            roles=roles,

            split_ratios=args.split_ratios,

            split_seed=args.split_seed,

            manual_layers=args.layers,

            fixed_high_count=args.fixed_high_count,

            bsi_threshold=args.bsi_threshold,

            max_dynamic_layers=args.max_dynamic_layers,

            stability_check=not args.no_stability_check,

            clean_quantile=args.clean_quantile,

            suspicious_quantile=args.suspicious_quantile,

            use_labels_for_f1=not args.no_label_threshold_refine,

        )



        if ids != feature_ids:

            raise RuntimeError(

                "Internal BYE id order changed unexpectedly."

            )



        print(

            f"BYE selected layers: {layers}"

        )



        print("Layer diagnostics:")

        for item in diagnostics:

            print(

                f"  layer={item['layer']} "

                f"BSI={item['bsi']:.6f} "

                f"minority_weight={item['target_weight']:.6f}"

            )



        counts, report = print_report(
            method="BYE",
            scores=scores,
            roles=roles,
            threshold=threshold,
        )

        if args.save:
            result = {
                "method": "bye",
                "num_samples": int(len(ids)),
                "eval_samples": int(len(ids)),
                "counts": counts,
                "parameters": {
                    "split_ratios": [float(v) for v in args.split_ratios],
                    "split_seed": int(args.split_seed),
                    "manual_layers": (
                        [int(v) for v in args.layers]
                        if args.layers is not None
                        else None
                    ),
                    "fixed_high_count": int(args.fixed_high_count),
                    "bsi_threshold": float(args.bsi_threshold),
                    "max_dynamic_layers": (
                        int(args.max_dynamic_layers)
                        if args.max_dynamic_layers is not None
                        else None
                    ),
                    "stability_check": bool(not args.no_stability_check),
                    "clean_quantile": float(args.clean_quantile),
                    "suspicious_quantile": float(args.suspicious_quantile),
                    "label_threshold_refine": bool(
                        not args.no_label_threshold_refine
                    ),
                },
                "decision": {
                    "threshold": float(threshold),
                },
                "metrics": {
                    "bye": report,
                },
                "diagnostics": {
                    "selected_layers": [int(v) for v in layers],
                    "layer_diagnostics": diagnostics,
                    "split_sizes": {
                        name: int(len(indexes))
                        for name, indexes in splits.items()
                    },
                },
            }
            save_json(args.save, result)
            print(f"\nSaved result JSON: {args.save}")

        if args.save_scores:

            save_score_cache(

                path=args.save_scores,

                method="bye",

                ids=ids,

                roles=roles,

                scores=scores,

                threshold=threshold,

                bye_layers=layers,

                bye_diagnostics=diagnostics,

                bye_splits=splits,

            )



    else:

        raise ValueError(

            f"Unsupported method: {method}"

        )





if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(

            "Unified evaluation for offline TCAP/BYE features. "

            "Prints P/R/F1 and AUC(B/C), AUC(B/I), AUC(B/T). "

            "Optionally saves summary JSON and per-sample scores as NPZ."

        )

    )



    # --------------------------------------------------------

    # Required inputs

    # --------------------------------------------------------



    parser.add_argument(

        "--features",

        type=str,

        required=True,

        help=(

            "tcap_features.jsonl or bye_features.jsonl."

        ),

    )



    parser.add_argument(

        "--labels",

        type=str,

        required=True,

        help=(

            "Dataset JSON/JSONL containing id/question_id "

            "and the C/I/T/B label field."

        ),

    )



    parser.add_argument(

        "--method",

        choices=["auto", "tcap", "bye"],

        default="auto",

        help=(

            "Default auto: infer detector from feature schema."

        ),

    )



    # --------------------------------------------------------

    # Label mapping

    # --------------------------------------------------------



    parser.add_argument(

        "--label-field",

        type=str,

        default="poison_role",

    )



    parser.add_argument(

        "--c-values",

        type=str,

        nargs="+",

        default=None,

        help=(

            "Label values treated as C. "

            "Default recognizes clean/negative_clean/etc."

        ),

    )



    parser.add_argument(

        "--i-values",

        type=str,

        nargs="+",

        default=None,

        help=(

            "Label values treated as I. "

            "Default recognizes image_only/negative_image/etc."

        ),

    )



    parser.add_argument(

        "--t-values",

        type=str,

        nargs="+",

        default=None,

        help=(

            "Label values treated as T. "

            "Default recognizes text_only/negative_text/etc."

        ),

    )



    parser.add_argument(

        "--b-values",

        type=str,

        nargs="+",

        default=None,

        help=(

            "Label values treated as B. "

            "Default recognizes positive_both/both/etc."

        ),

    )



    # --------------------------------------------------------

    # Optional reusable per-sample score cache

    # --------------------------------------------------------



    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Optional JSON path for the evaluation summary "
            "(P/R/F1, pairwise AUC, threshold, parameters, diagnostics)."
        ),
    )


    parser.add_argument(

        "--save-scores",

        type=str,

        default=None,

        help=(

            "Optional .npz path for ids/roles/continuous score/prediction "

            "and method-specific diagnostics. Default: print only."

        ),

    )



    # --------------------------------------------------------

    # TCAP parameters

    # --------------------------------------------------------



    parser.add_argument(

        "--tcap-last-layers",

        type=int,

        default=8,

    )



    parser.add_argument(

        "--tcap-top-heads",

        type=int,

        default=10,

    )



    parser.add_argument(

        "--tcap-vote-threshold",

        type=float,

        default=1e-4,

    )



    parser.add_argument(

        "--tcap-part",

        choices=["sys", "img", "usr"],

        default="sys",

    )



    parser.add_argument(

        "--n-jobs",

        type=int,

        default=-1,

    )



    # --------------------------------------------------------

    # BYE parameters

    # --------------------------------------------------------



    parser.add_argument(

        "--split-ratios",

        type=float,

        nargs=3,

        default=[0.6, 0.2, 0.2],

        metavar=(

            "SELECTION",

            "CALIBRATION",

            "TEST",

        ),

    )



    parser.add_argument(

        "--split-seed",

        type=int,

        default=42,

    )



    parser.add_argument(

        "--layers",

        type=int,

        nargs="+",

        default=None,

        help=(

            "Manual BYE layer indices. "

            "Default: fixed-high + BSI dynamic."

        ),

    )



    parser.add_argument(

        "--fixed-high-count",

        type=int,

        default=8,

    )



    parser.add_argument(

        "--bsi-threshold",

        type=float,

        default=2.0,

    )



    parser.add_argument(

        "--max-dynamic-layers",

        type=int,

        default=4,

    )



    parser.add_argument(

        "--no-stability-check",

        action="store_true",

    )



    parser.add_argument(

        "--clean-quantile",

        type=float,

        default=0.5,

    )



    parser.add_argument(

        "--suspicious-quantile",

        type=float,

        default=0.95,

    )



    parser.add_argument(

        "--no-label-threshold-refine",

        action="store_true",

        help=(

            "Do not use calibration labels to refine the BYE "

            "suspicious threshold; use quantile only."

        ),

    )



    args = parser.parse_args()



    main(args)
