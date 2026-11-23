#!/usr/bin/env python
"""Pure-NumPy smoke test for the Lastde math (no torch, no downloads).

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.core import (  # noqa: E402
    auroc,
    cosine_adjacent,
    diversity_entropy,
    lastde_plus_plus,
    lastde_score,
    moving_average_scales,
    sliding_windows,
)


def approx(a, b, tol=1e-6, msg=""):
    assert abs(float(a) - float(b)) <= tol, f"{msg}: {a} != {b}"


def main() -> None:
    # 1) multi-scale moving average
    scales = moving_average_scales([-1, -2, -3, -4], 2)
    assert set(scales) == {1, 2}
    assert np.allclose(scales[1], [-1, -2, -3, -4])
    assert np.allclose(scales[2], [-1.5, -2.5, -3.5])

    # 2) sliding windows
    mat = sliding_windows([1, 2, 3, 4], 3)
    assert mat.shape == (2, 3)
    assert np.allclose(mat, [[1, 2, 3], [2, 3, 4]])

    # 3) cosine of orthogonal rows
    sims = cosine_adjacent(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
    assert np.allclose(sims, [0.0, 0.0])

    # 4) DE: concentrated -> 0, uniform across 10 bins -> 1
    approx(diversity_entropy([0.5] * 10, 10), 0.0, msg="concentrated DE")
    approx(diversity_entropy(np.linspace(-0.9, 0.9, 10), 10), 1.0, tol=1e-9, msg="uniform DE")

    # 5) Lastde on the 7-token running example
    tps = np.array([-1.20, -3.60, -1.00, -2.80, -0.80, -4.20, -1.40])
    res = lastde_score(tps, s=3, eps=10, tau_prime=3)
    approx(res.log_likelihood, -2.142857, tol=1e-5, msg="LL")
    approx(res.des[0], 0.2442, tol=0.01, msg="DE tau=1")
    approx(res.des[1], 0.0, tol=1e-9, msg="DE tau=2")
    approx(res.des[2], 0.0, tol=1e-9, msg="DE tau=3")
    approx(res.agg_mde, 0.1150, tol=1e-3, msg="Agg-MDE std")
    print(f"Lastde example: score={res.score:.3f}, LL={res.log_likelihood:.3f}, "
          f"DEs={[round(x, 3) for x in res.des]}, Agg={res.agg_mde:.3f}")

    # 6) AUROC: known 4-vs-4 example -> 13/16
    a = auroc([0.3, 0.5, 0.8, 0.9], [0.1, 0.2, 0.4, 0.7])
    approx(a, 0.8125, msg="AUROC")

    # 7) Lastde++ normalization
    z, base, sample_scores = lastde_plus_plus(
        tps, [tps + 0.1, tps - 0.1], s=3, eps=10, tau_prime=3
    )
    assert np.isfinite(z)
    assert len(sample_scores) == 2

    print("ALL CORE CHECKS PASSED")


if __name__ == "__main__":
    main()
