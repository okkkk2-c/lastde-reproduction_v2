"""Baseline statistics.

Convention: every function returns a score where **higher = more likely AI**.
The four cheap baselines follow the sample-based group of the paper:
Likelihood, LogRank, Entropy, DetectLRR.  Fast-DetectGPT-style sampling
discrepancy is also provided.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .models import ModelOutput


def likelihood(out: ModelOutput) -> float:
    return float(np.mean(out.tps))


def logrank(out: ModelOutput) -> float:
    if out.ranks is None:
        raise ValueError("ranks missing; use a model wrapper that returns them")
    # lower rank = more likely = more AI-like; negative log rank keeps higher=AI
    return float(-np.mean(np.log(np.maximum(out.ranks, 1))))


def entropy(out: ModelOutput) -> float:
    if out.entropies is None:
        raise ValueError("entropies missing")
    # lower predictive entropy = more AI-like
    return float(-np.mean(out.entropies))


def detectlrr(out: ModelOutput) -> float:
    """Ratio of log-likelihood to log-rank (DetectLLM)."""
    if out.ranks is None:
        raise ValueError("ranks missing")
    mean_logp = float(np.mean(out.tps))
    mean_logrank = float(np.mean(np.log(np.maximum(out.ranks, 1))))
    # Convention: higher = more AI.
    # raw LL/logrank is negative and actually ranks human higher, so flip the sign.
    return float(-mean_logp / max(mean_logrank, 1e-8))


def sampling_discrepancy(
    original: ModelOutput, samples: Sequence[ModelOutput]
) -> float:
    """Fast-DetectGPT-style score: (LL(x) - mean LL(samples)) / std."""
    if not samples:
        return 0.0
    ll = float(np.mean(original.tps))
    sample_lls = np.asarray(
        [float(np.mean(s.tps)) for s in samples if s.tps.size > 0]
    )
    if sample_lls.size == 0:
        return 0.0
    mu = float(sample_lls.mean())
    sigma = float(sample_lls.std(ddof=0))
    return float((ll - mu) / max(sigma, 1e-8))
