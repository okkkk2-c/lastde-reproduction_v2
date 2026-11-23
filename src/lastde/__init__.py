"""Lastde: training-free LLM text detection reproduction package."""
from .core import (
    LastdeResult,
    aggregate,
    auroc,
    bootstrap_auroc_ci,
    cosine_adjacent,
    diversity_entropy,
    eps_for,
    lastde_plus_plus,
    lastde_score,
    moving_average_scales,
    sliding_windows,
)

__all__ = [
    "LastdeResult",
    "aggregate",
    "auroc",
    "bootstrap_auroc_ci",
    "cosine_adjacent",
    "diversity_entropy",
    "eps_for",
    "lastde_plus_plus",
    "lastde_score",
    "moving_average_scales",
    "sliding_windows",
]
