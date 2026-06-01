"""Hypothesis-test wrappers returning structured results, plus FDR correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    alpha: float = config.ALPHA

    @property
    def significant(self) -> bool:
        return self.pvalue < self.alpha


def mann_whitney(a, b, alternative: str = "two-sided") -> TestResult:
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return TestResult(f"Mann-Whitney U ({alternative})", float(u), float(p))


def ks_test(a, b) -> TestResult:
    s, p = stats.ks_2samp(a, b)
    return TestResult("Kolmogorov-Smirnov", float(s), float(p))


def levene_test(*groups) -> TestResult:
    s, p = stats.levene(*groups)
    return TestResult("Levene", float(s), float(p))


def kruskal_test(*groups) -> TestResult:
    s, p = stats.kruskal(*groups)
    return TestResult("Kruskal-Wallis", float(s), float(p))


def chi2_test(table) -> TestResult:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return TestResult(f"Chi-square (dof={dof})", float(chi2), float(p))


def fdr_bh(pvalues, alpha: float = config.ALPHA):
    """Benjamini-Hochberg FDR. Returns (reject: bool ndarray, p_adjusted: ndarray)."""
    pvalues = np.asarray(pvalues, float)
    reject, p_adj, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return reject, p_adj
