"""Effect-size estimators and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

from . import config


def cohen_d(a, b) -> float:
    """Cohen's d with pooled SD (ddof=1). Positive when mean(a) > mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled)


def variance_ratio(a, b) -> float:
    """Ratio of variances (a/b) — a spread effect size for Q2/Q3."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a.var(ddof=1) / b.var(ddof=1))


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from a Mann-Whitney U statistic."""
    return float((2.0 * u) / (n1 * n2) - 1.0)


def bootstrap_ci(stat_fn, a, b, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = config.RANDOM_SEED):
    """Percentile bootstrap CI for any two-sample statistic ``stat_fn(a, b)``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        stats[i] = stat_fn(sa, sb)
    lo = np.percentile(stats, 100 * (1 - ci) / 2)
    hi = np.percentile(stats, 100 * (1 + ci) / 2)
    return lo, hi


def cohen_d_ci(a, b, n_boot: int = 10000, ci: float = 0.95,
               seed: int = config.RANDOM_SEED):
    """Point estimate + percentile bootstrap CI for Cohen's d."""
    d = cohen_d(a, b)
    lo, hi = bootstrap_ci(cohen_d, a, b, n_boot=n_boot, ci=ci, seed=seed)
    return d, lo, hi
