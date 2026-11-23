"""Core mathematics for Lastde.

Pipeline:  TPS -> multi-scale -> sliding windows -> cosine similarity
           -> histogram -> diversity entropy (DE) -> aggregate -> Lastde score.

All functions are pure NumPy so they can be unit-tested without torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import math

import numpy as np


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass
class LastdeResult:
    score: float
    log_likelihood: float
    des: List[float]
    agg_mde: float
    n_tokens: int
    n_scales: int

    @property
    def perplexity(self) -> float:
        return float(math.exp(-self.log_likelihood))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_1d(tps: Sequence[float]) -> np.ndarray:
    arr = np.asarray(tps, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("TPS is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("TPS contains NaN/Inf")
    return arr


def moving_average_scales(tps: Sequence[float], tau_prime: int) -> Dict[int, np.ndarray]:
    """Eq. (2): moving average of tau consecutive log-probs, stride 1.

    Returns {tau: sequence of length n - tau + 1} for tau = 1..tau_prime.
    """
    tps = _as_1d(tps)
    n = tps.size
    if tau_prime < 1:
        raise ValueError("tau_prime must be >= 1")
    csum = np.concatenate(([0.0], np.cumsum(tps)))
    out: Dict[int, np.ndarray] = {}
    for tau in range(1, tau_prime + 1):
        if tau > n:
            break
        seq = (csum[tau:] - csum[:-tau]) / float(tau)
        out[tau] = seq
    return out


def sliding_windows(seq: Sequence[float], s: int) -> np.ndarray:
    """Stack windows of width s, stride 1. Shape: (n - s + 1, s)."""
    seq = np.asarray(seq, dtype=np.float64).reshape(-1)
    if s < 1:
        raise ValueError("s must be >= 1")
    if seq.size < s:
        return np.empty((0, s), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(seq, s).copy()


def cosine_adjacent(mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between adjacent rows of a (m, s) matrix."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    a, b = mat[:-1], mat[1:]
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sims = num / np.clip(den, 1e-12, None)
    return np.clip(sims, -1.0, 1.0)


def diversity_entropy(sims: Sequence[float], eps: int) -> float:
    """Eq. (8): normalized Shannon entropy of the similarity histogram.

    sims are binned into `eps` equal-width bins on [-1, 1].  DE in [0, 1].
    """
    if eps < 2:
        raise ValueError("eps must be >= 2")
    sims = np.asarray(sims, dtype=np.float64).reshape(-1)
    if sims.size == 0:
        return 0.0
    counts, _ = np.histogram(sims, bins=eps, range=(-1.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(eps))


def aggregate(des: Sequence[float], mode: str = "std") -> float:
    """Aggregate per-scale DE values into Agg-MDE."""
    arr = np.asarray(des, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    mode = mode.lower()
    if mode in ("std", "std0", "population_std"):
        # the paper reports Std; 0.115 for (0.244, 0, 0) is ddof=0
        return float(np.std(arr, ddof=0))
    if mode in ("expstd", "exp_std", "exp-std"):
        # official README recommends ExpStd: exp(std(DE))
        return float(np.exp(np.std(arr, ddof=0)))
    if mode in ("sum",):
        return float(arr.sum())
    if mode in ("mean",):
        return float(arr.mean())
    if mode in ("range", "max-min"):
        return float(arr.max() - arr.min())
    if mode in ("2norm", "l2"):
        return float(np.linalg.norm(arr))
    raise ValueError(f"unknown aggregate mode: {mode}")


def eps_for(n_tokens: int, factor: float = 10.0, minimum: int = 10) -> int:
    """Paper default: eps = factor * n (e.g. 10n); keep it an int >= 2."""
    return max(minimum, int(round(factor * n_tokens)))


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def lastde_score(
    tps: Sequence[float],
    s: int = 3,
    eps: int = 10,
    tau_prime: int = 5,
    agg: str = "std",
) -> LastdeResult:
    """Lastde = mean(TPS) / Agg(DE_1, ..., DE_tau')."""
    tps = _as_1d(tps)
    scales = moving_average_scales(tps, tau_prime)
    des: List[float] = []
    for tau in sorted(scales):
        mat = sliding_windows(scales[tau], s)
        if mat.shape[0] < 2:
            des.append(0.0)
            continue
        sims = cosine_adjacent(mat)
        des.append(diversity_entropy(sims, eps))
    ll = float(np.mean(tps))
    agg_mde = aggregate(des, agg)
    # guard: std can be exactly zero for very short / degenerate texts
    score = ll / max(agg_mde, 1e-8)
    return LastdeResult(
        score=float(score),
        log_likelihood=ll,
        des=des,
        agg_mde=agg_mde,
        n_tokens=int(tps.size),
        n_scales=len(scales),
    )


def lastde_plus_plus(
    tps: Sequence[float],
    sample_tps: Sequence[Sequence[float]],
    s: int = 4,
    eps: int = 80,
    tau_prime: int = 15,
    agg: str = "std",
) -> Tuple[float, LastdeResult, List[float]]:
    """Eq. (11): z-normalize the Lastde score against sampled continuations."""
    base = lastde_score(tps, s=s, eps=eps, tau_prime=tau_prime, agg=agg)
    sample_scores = [
        lastde_score(x, s=s, eps=eps, tau_prime=tau_prime, agg=agg).score
        for x in sample_tps
        if len(x) > 0
    ]
    if not sample_scores:
        return base.score, base, []
    mu = float(np.mean(sample_scores))
    sigma = float(np.std(sample_scores, ddof=0))
    z = (base.score - mu) / max(sigma, 1e-8)
    return float(z), base, sample_scores


# ---------------------------------------------------------------------------
# AUROC (rank based, tie aware)
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 * P(tie); higher score = more AI."""
    pos = np.asarray(positive, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both classes must be non-empty")
    ranks = _rankdata(np.concatenate([pos, neg]))
    n_pos, n_neg = pos.size, neg.size
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_auroc_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for AUROC (useful with 150 pairs)."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=pos.size, replace=True)
        n = rng.choice(neg, size=neg.size, replace=True)
        vals.append(auroc(p, n))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
