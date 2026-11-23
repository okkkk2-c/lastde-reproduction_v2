"""Evaluation utilities: AUROC per method, bootstrap CI, summary table."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .core import auroc, bootstrap_auroc_ci


def evaluate_rows(
    rows: Sequence[Dict],
    n_boot: int = 500,
    seed: int = 0,
) -> List[Dict]:
    """`rows` contain dataset/model/pair_id/label/method/score."""
    methods = sorted({r["method"] for r in rows})
    keys = sorted({(r["dataset"], r["model"]) for r in rows})
    summary: List[Dict] = []
    for dataset, model in keys:
        for method in methods:
            pos = [
                r["score"]
                for r in rows
                if r["dataset"] == dataset
                and r["model"] == model
                and r["method"] == method
                and r["label"] == "llm"
            ]
            neg = [
                r["score"]
                for r in rows
                if r["dataset"] == dataset
                and r["model"] == model
                and r["method"] == method
                and r["label"] == "human"
            ]
            if len(pos) < 2 or len(neg) < 2:
                continue
            a = auroc(pos, neg)
            if n_boot and n_boot > 0:
                lo, hi = bootstrap_auroc_ci(pos, neg, n_boot=n_boot, seed=seed)
            else:
                lo, hi = float("nan"), float("nan")
            summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "auroc": a,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_pairs": len(pos),
                }
            )
    return summary


def summary_dataframe(summary: Sequence[Dict]):
    import pandas as pd

    df = pd.DataFrame(summary)
    if df.empty:
        return df
    return df.pivot_table(
        index=["dataset", "model"], columns="method", values="auroc"
    ).round(4)


def format_auroc(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "  n/a"
    return f"{value * 100:6.2f}"
