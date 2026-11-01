# Stage 2: offline feature derivation for TCAP and BYE.
#
# Reads:
#   raw_dir/
#   ├── raw_index.jsonl
#   └── attn/
#       ├── *.npz
#
# Each npz contains:
#   attn_vec   : [L, H, T]
#   token_roles: [T]
#
# Derives:
#   --method tcap
#       tri-component attention ratios:
#       system / image / user
#       -> tcap_features.jsonl
#
#   --method bye
#       normalized Shannon entropy over image-token attention
#       -> bye_features.jsonl
#
# Both methods are derived from exactly the same Stage-1 raw attention data.
# No model forward pass is required.

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


# ============================================================
# Token roles stored by Stage-1
# ============================================================

ROLE_SYSTEM = 0
ROLE_IMAGE = 1
ROLE_USER = 2
ROLE_QUERY = 3

EPS = 1e-300


# ============================================================
# Load raw metadata
# ============================================================

def load_raw_records(raw_dir):
    index_path = os.path.join(
        raw_dir,
        "raw_index.jsonl",
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Raw index not found: {index_path}"
        )

    records = []

    with open(
        index_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    if not records:
        raise ValueError(
            f"Raw index is empty: {index_path}"
        )

    return records


# ============================================================
# Load one attention npz
# ============================================================

def load_raw_arrays(raw_dir, meta):

    path = os.path.join(
        raw_dir,
        meta["file"],
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw attention array not found: {path}"
        )

    with np.load(path) as data:

        attn_vec = data[
            "attn_vec"
        ].astype(np.float64)

        token_roles = data[
            "token_roles"
        ]

    # --------------------------------------------------------
    # Shape sanity checks
    # --------------------------------------------------------

    if attn_vec.ndim != 3:
        raise ValueError(
            f"Invalid attention shape in {path}: "
            f"expected [L,H,T], got {attn_vec.shape}"
        )

    if token_roles.ndim != 1:
        raise ValueError(
            f"Invalid token_roles shape in {path}: "
            f"expected [T], got {token_roles.shape}"
        )

    if token_roles.shape[0] != attn_vec.shape[-1]:
        raise ValueError(
            f"Token length mismatch in {path}: "
            f"attn={attn_vec.shape}, "
            f"roles={token_roles.shape}"
        )

    return attn_vec, token_roles


# ============================================================
# Tri-component partition for TCAP
# ============================================================

def component_positions(
    token_roles,
    fold_query_to,
):

    system = np.where(
        token_roles == ROLE_SYSTEM
    )[0]

    image = np.where(
        token_roles == ROLE_IMAGE
    )[0]

    user = np.where(
        token_roles == ROLE_USER
    )[0]

    query = np.where(
        token_roles == ROLE_QUERY
    )[0]

    # --------------------------------------------------------
    # InternVL may explicitly mark the decode query position.
    #
    # TCAP requires exactly three components:
    #   sys / img / usr
    #
    # Therefore query tokens must be folded into one component.
    # --------------------------------------------------------

    if fold_query_to == "system":

        system = np.concatenate(
            [system, query]
        )

    elif fold_query_to == "user":

        user = np.concatenate(
            [user, query]
        )

    else:
        raise ValueError(
            "fold_query_to must be "
            "'system' or 'user'."
        )

    # --------------------------------------------------------
    # Ensure every token belongs to exactly one final component
    # --------------------------------------------------------

    covered = np.concatenate([
        system,
        image,
        user,
    ])

    if covered.shape[0] != token_roles.shape[0]:
        raise ValueError(
            "Component partition does not "
            "cover the whole sequence."
        )

    return {
        "sys": system,
        "img": image,
        "usr": user,
    }


# ============================================================
# TCAP feature derivation
# ============================================================

def derive_tcap(
    meta,
    attn_vec,
    token_roles,
    fold_query_to,
):

    """
    TCAP feature:

        attention [L,H,T]
            ↓
        sum attention mass over:
            system / image / user
            ↓
        ratio [L,H,3]

    Component order:
        0 -> sys
        1 -> img
        2 -> usr
    """

    comp = component_positions(
        token_roles,
        fold_query_to,
    )

    # Total attention mass for every layer/head
    # [L,H,T] -> [L,H]

    total = attn_vec.sum(
        axis=-1
    )

    total = np.clip(
        total,
        EPS,
        None,
    )

    ratios = np.stack(
        [

            attn_vec[
                :, :, comp["sys"]
            ].sum(axis=-1) / total,

            attn_vec[
                :, :, comp["img"]
            ].sum(axis=-1) / total,

            attn_vec[
                :, :, comp["usr"]
            ].sum(axis=-1) / total,

        ],
        axis=-1,
    )

    # [L,H,3]

    return {

        "question_id":
            meta["question_id"],

        "answer":
            meta.get("answer", ""),

        "num_tokens": (
            int(comp["sys"].shape[0]),
            int(comp["img"].shape[0]),
            int(comp["usr"].shape[0]),
        ),

        "tcap_map":
            ratios.tolist(),
    }


# ============================================================
# Normalized Shannon entropy
# ============================================================

def normalized_shannon(values):

    """
    values:
        [..., N]

    First renormalize attention within the selected component:

        p_i = a_i / sum_j a_j

    Then compute:

        H = -sum p_i log(p_i)

    Normalized:

        H_norm = H / log(N)

    Therefore approximately:

        H_norm -> 0
            attention highly concentrated

        H_norm -> 1
            attention approximately uniform
    """

    n = values.shape[-1]

    if n < 2:
        raise ValueError(
            f"Normalized Shannon entropy "
            f"requires at least 2 tokens, got {n}."
        )

    totals = values.sum(
        axis=-1,
        keepdims=True,
    )

    probabilities = (
        values /
        np.clip(
            totals,
            EPS,
            None,
        )
    )

    entropy = -(
        probabilities
        *
        np.log(
            np.clip(
                probabilities,
                EPS,
                1.0,
            )
        )
    ).sum(axis=-1)

    return entropy / np.log(n)


# ============================================================
# BYE feature derivation
# ============================================================

def derive_bye(
    meta,
    attn_vec,
    token_roles,
):

    """
    BYE feature:

        attention [L,H,T]
            ↓
        keep image tokens
            ↓
        [L,H,N_img]
            ↓
        normalized Shannon entropy
            ↓
        [L,H]
            ↓
        average over heads
            ↓
        [L]

    entropy_by_layer_head:
        [L,H]

    entropy_by_layer:
        [L]
    """

    image_positions = np.where(
        token_roles == ROLE_IMAGE
    )[0]

    num_image_tokens = int(
        image_positions.shape[0]
    )

    if num_image_tokens < 2:
        raise ValueError(
            f"Question {meta.get('question_id')} "
            f"has only {num_image_tokens} image tokens."
        )

    # --------------------------------------------------------
    # Select image-token attention
    #
    # [L,H,T] -> [L,H,N_img]
    # --------------------------------------------------------

    visual_values = attn_vec[
        :,
        :,
        image_positions,
    ]

    # --------------------------------------------------------
    # Per-layer, per-head entropy
    #
    # [L,H,N_img] -> [L,H]
    # --------------------------------------------------------

    entropy_by_layer_head = (
        normalized_shannon(
            visual_values
        )
    )

    # --------------------------------------------------------
    # BYE layer signal
    #
    # average over heads:
    #
    # [L,H] -> [L]
    # --------------------------------------------------------

    entropy_by_layer = (
        entropy_by_layer_head.mean(
            axis=1
        )
    )

    return {

        "question_id":
            meta["question_id"],

        "answer":
            meta.get("answer", ""),

        "num_layers":
            int(
                entropy_by_layer_head.shape[0]
            ),

        "num_heads":
            int(
                entropy_by_layer_head.shape[1]
            ),

        "image_token_count":
            num_image_tokens,

        "entropy_by_layer":
            entropy_by_layer.tolist(),

        "entropy_by_layer_head":
            entropy_by_layer_head.tolist(),
    }


# ============================================================
# JSONL writer
# ============================================================

def write_jsonl(
    path,
    rows,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Main derivation
# ============================================================

def run(args):

    raw_records = load_raw_records(
        args.raw_dir
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    methods = args.method

    if "all" in methods:
        methods = [
            "tcap",
            "bye",
        ]

    print(
        f"Loaded {len(raw_records)} raw records "
        f"from {args.raw_dir}"
    )

    print(
        f"Methods: {methods}"
    )

    outputs = {
        method: []
        for method in methods
    }

    # ========================================================
    # Process every sample
    # ========================================================

    for meta in tqdm(
        raw_records,
        desc="Deriving TCAP/BYE features",
        disable=None
    ):

        attn_vec, token_roles = (
            load_raw_arrays(
                args.raw_dir,
                meta,
            )
        )

        # ----------------------------------------------------
        # TCAP
        # ----------------------------------------------------

        if "tcap" in outputs:

            tcap_record = derive_tcap(
                meta,
                attn_vec,
                token_roles,
                args.fold_query_to,
            )

            outputs["tcap"].append(
                tcap_record
            )

        # ----------------------------------------------------
        # BYE
        # ----------------------------------------------------

        if "bye" in outputs:

            bye_record = derive_bye(
                meta,
                attn_vec,
                token_roles,
            )

            outputs["bye"].append(
                bye_record
            )

        del attn_vec
        del token_roles

    # ========================================================
    # Write output files
    # ========================================================

    file_names = {

        "tcap":
            "tcap_features.jsonl",

        "bye":
            "bye_features.jsonl",
    }

    for method in methods:

        output_path = os.path.join(
            args.output_dir,
            file_names[method],
        )

        write_jsonl(
            output_path,
            outputs[method],
        )

        print(
            f"Wrote {len(outputs[method])} "
            f"{method} feature records "
            f"to {output_path}"
        )

    # ========================================================
    # Sanity report
    # ========================================================

    if (
        "tcap" in outputs
        and outputs["tcap"]
    ):

        first = outputs["tcap"][0]

        tcap_map = first[
            "tcap_map"
        ]

        num_layers = len(
            tcap_map
        )

        num_heads = (
            len(tcap_map[0])
            if tcap_map
            else 0
        )

        print(
            "\n[TCAP sanity]"
        )

        print(
            f"question_id: "
            f"{first['question_id']}"
        )

        print(
            f"num_tokens "
            f"(sys,img,usr): "
            f"{first['num_tokens']}"
        )

        print(
            f"tcap_map shape: "
            f"[L={num_layers}]"
            f"[H={num_heads}]"
            f"[3]"
        )

    if (
        "bye" in outputs
        and outputs["bye"]
    ):

        first = outputs["bye"][0]

        num_layers = len(
            first["entropy_by_layer"]
        )

        num_heads = (
            len(
                first[
                    "entropy_by_layer_head"
                ][0]
            )
            if first[
                "entropy_by_layer_head"
            ]
            else 0
        )

        print(
            "\n[BYE sanity]"
        )

        print(
            f"question_id: "
            f"{first['question_id']}"
        )

        print(
            f"image_token_count: "
            f"{first['image_token_count']}"
        )

        print(
            f"entropy_by_layer shape: "
            f"[L={num_layers}]"
        )

        print(
            f"entropy_by_layer_head shape: "
            f"[L={num_layers}]"
            f"[H={num_heads}]"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Offline TCAP and BYE feature derivation "
            "from Stage-1 raw attention."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help=(
            "Stage-1 output directory containing "
            "raw_index.jsonl and attn/*.npz."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,

        help=(
            "Directory for "
            "tcap_features.jsonl and "
            "bye_features.jsonl."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=["tcap"],
        choices=[
            "all",
            "tcap",
            "bye",
        ],
        help=(
            "Feature type to derive. "
            "Default: both TCAP and BYE."
        ),
    )

    parser.add_argument(
        "--fold-query-to",
        type=str,
        default="system",
        choices=[
            "system",
            "user",
        ],
        help=(
            "For TCAP only: which tri-component owns "
            "ROLE_QUERY tokens. "
            "InternVL may contain explicit query-role tokens; "
            "LLaVA usually does not."
        ),
    )

    args = parser.parse_args()

    run(args)