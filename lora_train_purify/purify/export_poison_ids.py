#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path

import numpy as np


def decode_ids(arr):
    """
    将不同 numpy 字符串类型统一转换成 Python str。
    """
    result = []

    for x in arr:
        if isinstance(x, bytes):
            result.append(x.decode("utf-8"))
        else:
            result.append(str(x))

    return np.asarray(result, dtype=str)


def select_score_key(data):
    """
    自动选择 detector score 字段。

    优先级：
        final_score
        tcap_score
        bye_score
        score
    """

    candidates = [
        "final_score",
        "tcap_score",
        "bye_score",
        "score",
    ]

    for key in candidates:
        if key in data.files:
            return key

    raise ValueError(
        "未找到分数字段。支持字段："
        "final_score / tcap_score / bye_score / score\n"
        f"当前 NPZ 字段：{data.files}"
    )


def scalar_to_python(x):
    """
    numpy scalar -> Python scalar / str
    """
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()

    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, np.generic):
        return x.item()

    return x


def main(args):

    npz_path = Path(args.scores)

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(
        npz_path,
        allow_pickle=False,
    )

    print("=" * 70)
    print(f"Input NPZ : {npz_path}")
    print(f"NPZ keys  : {data.files}")
    print("=" * 70)

    # =========================================================
    # 1. 基础字段
    # =========================================================

    if "ids" not in data.files:
        raise ValueError(
            f"缺少 ids 字段，当前字段：{data.files}"
        )

    ids = decode_ids(data["ids"])

    n_total = len(ids)

    # =========================================================
    # 2. eval_mask
    # =========================================================

    if "eval_mask" in data.files:
        eval_mask = np.asarray(
            data["eval_mask"],
            dtype=bool,
        )

        if len(eval_mask) != n_total:
            raise ValueError(
                "eval_mask 长度和 ids 不一致："
                f"{len(eval_mask)} vs {n_total}"
            )
    else:
        eval_mask = np.ones(
            n_total,
            dtype=bool,
        )

    # =========================================================
    # 3. score
    # =========================================================

    score_key = select_score_key(data)

    scores = np.asarray(
        data[score_key],
        dtype=float,
    )

    if len(scores) != n_total:
        raise ValueError(
            f"{score_key} 长度和 ids 不一致："
            f"{len(scores)} vs {n_total}"
        )

    # =========================================================
    # 4. prediction
    # =========================================================

    if "pred" in data.files:

        pred = np.asarray(
            data["pred"],
            dtype=bool,
        )

        pred_source = "pred"

    else:

        # 如果没有 pred，就尝试 threshold + score
        if "threshold" in data.files:

            threshold = float(
                scalar_to_python(
                    data["threshold"]
                )
            )

        else:

            threshold = args.threshold

        if threshold is None:
            raise ValueError(
                "NPZ 中既没有 pred，也没有 threshold；"
                "请通过 --threshold 指定阈值。"
            )

        pred = scores >= threshold
        pred_source = (
            f"{score_key} >= {threshold}"
        )

    if len(pred) != n_total:
        raise ValueError(
            "pred 长度和 ids 不一致："
            f"{len(pred)} vs {n_total}"
        )

    # =========================================================
    # 5. 只在 eval_mask 有效区域中导出
    # =========================================================

    valid_mask = eval_mask

    poison_mask = (
        pred
        &
        valid_mask
    )

    poison_ids = ids[poison_mask]

    poison_scores = scores[poison_mask]

    # =========================================================
    # 6. method / threshold 元信息
    # =========================================================

    if "method" in data.files:
        method = scalar_to_python(
            data["method"]
        )
    else:
        method = "unknown"

    if "threshold" in data.files:
        threshold = float(
            scalar_to_python(
                data["threshold"]
            )
        )
    else:
        threshold = args.threshold

    # =========================================================
    # 7. JSON
    # =========================================================

    result = {

        "source":
            str(npz_path),

        "method":
            str(method),

        "score_key":
            score_key,

        "prediction_source":
            pred_source,

        "threshold":
            threshold,

        "total_samples":
            int(n_total),

        "evaluated_samples":
            int(valid_mask.sum()),

        "detected_poison_count":
            int(poison_mask.sum()),

        "poison_rate_total":
            float(
                poison_mask.sum()
                /
                max(n_total, 1)
            ),

        "poison_rate_evaluated":
            float(
                poison_mask.sum()
                /
                max(valid_mask.sum(), 1)
            ),

        "ids":
            poison_ids.tolist(),

        "scores":
            poison_scores.tolist(),

    }

    json_path = Path(
        args.output_json
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================
    # 8. TXT
    # =========================================================

    txt_path = Path(
        args.output_txt
    )

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        for sid in poison_ids:

            f.write(
                str(sid)
                +
                "\n"
            )

    # =========================================================
    # 9. 输出统计
    # =========================================================

    print()
    print("=" * 70)

    print(
        f"Method              : {method}"
    )

    print(
        f"Score key           : {score_key}"
    )

    print(
        f"Prediction source   : {pred_source}"
    )

    if threshold is not None:
        print(
            f"Threshold           : {threshold}"
        )

    print("-" * 70)

    print(
        f"Total samples       : {n_total}"
    )

    print(
        f"Evaluated samples   : {valid_mask.sum()}"
    )

    print(
        f"Detected poison     : {poison_mask.sum()}"
    )

    print(
        f"Poison rate (total) : "
        f"{poison_mask.sum()/max(n_total,1):.4f}"
    )

    print(
        f"Poison rate (eval)  : "
        f"{poison_mask.sum()/max(valid_mask.sum(),1):.4f}"
    )

    print("-" * 70)

    print(
        f"Saved JSON          : {json_path}"
    )

    print(
        f"Saved TXT           : {txt_path}"
    )

    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        required=True,
        help="Ours / TCAP / BYE 输出的 NPZ",
    )

    parser.add_argument(
        "--output-json",
        default="defense_results/poison_ids.json",
    )

    parser.add_argument(
        "--output-txt",
        default="defense_results/poison_ids.txt",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "当 NPZ 中没有 pred 时使用。"
            "如果已有 pred，则忽略。"
        ),
    )

    args = parser.parse_args()

    main(args)