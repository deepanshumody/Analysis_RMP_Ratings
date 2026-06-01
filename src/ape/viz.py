"""Plotting helpers.

Every function builds and returns a ``matplotlib.figure.Figure``; nothing is
shown or written to disk implicitly. ``save_figure`` is the only function that
touches the filesystem. This replaces the old ``plt.savefig(f"plot_{ts}.png")``
calls scattered through ``main.py`` that spammed the repo root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

MALE_COLOR = "#2c7fb8"
FEMALE_COLOR = "#d95f6b"
ACCENT = "#542c8b"


def density_plot(s1, s2, label1, label2, *, bins: int = 30, title: str | None = None) -> Figure:
    """Overlaid normalized histograms (KDE) for two groups."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(s1, bins=bins, kde=True, color=MALE_COLOR, label=label1, stat="density", ax=ax)
    sns.histplot(s2, bins=bins, kde=True, color=FEMALE_COLOR, label=label2, stat="density", ax=ax)
    ax.set(title=title or "", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    return fig


def ci_plot(series, *, title: str | None = None) -> Figure:
    """Histogram with mean and 95% CI of the mean marked."""
    series = pd.Series(series).dropna()
    mean = series.mean()
    half = 1.96 * series.std() / np.sqrt(len(series))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(series, bins=30, kde=True, color=MALE_COLOR, ax=ax)
    ax.axvline(mean - half, color="red", linestyle="--", label=f"Lower CI: {mean - half:.2f}")
    ax.axvline(mean, color="black", label=f"Mean: {mean:.2f}")
    ax.axvline(mean + half, color="green", linestyle="--", label=f"Upper CI: {mean + half:.2f}")
    ax.set(title=title or "", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    return fig


def bootstrap_effect_plot(values, lo, hi, *, title: str | None = None) -> Figure:
    """Bootstrap distribution of an effect size with its CI bounds marked."""
    values = np.asarray(values, float)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(values, bins=30, kde=True, color=ACCENT, ax=ax)
    ax.axvline(lo, color="red", linestyle="--", label=f"Lower: {lo:.3f}")
    ax.axvline(values.mean(), color="black", label=f"Mean: {values.mean():.3f}")
    ax.axvline(hi, color="green", linestyle="--", label=f"Upper: {hi:.3f}")
    ax.set(title=title or "", xlabel="Effect size", ylabel="Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def pval_scatter(df_pvals: pd.DataFrame, *, title: str | None = None) -> Figure:
    """Scatter of per-tag p-values (Mann-Whitney vs KS), labelled by tag."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pvals, x="mwu_p", y="ks_p", ax=ax)
    for _, row in df_pvals.iterrows():
        ax.text(row["mwu_p"], row["ks_p"], row["tag"], fontsize=7, rotation=30)
    ax.set(title=title or "P-values by tag", xlabel="Mann-Whitney U p-value",
           ylabel="KS test p-value")
    fig.tight_layout()
    return fig


def tag_significance_bar(labels, pvalues, *, alpha: float = 0.005,
                         title: str | None = None) -> Figure:
    """Horizontal bar chart of -log10(p) per tag, sorted, with a significance line."""
    pvals = np.asarray(pvalues, float)
    neg = -np.log10(np.clip(pvals, 1e-300, 1.0))
    order = np.argsort(neg)
    labels = [labels[i] for i in order]
    neg = neg[order]
    thr = -np.log10(alpha)
    colors = [MALE_COLOR if v >= thr else "#c2c2c2" for v in neg]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, neg, color=colors)
    ax.axvline(thr, color="red", linestyle="--", label=f"alpha = {alpha} (FDR)")
    ax.set(title=title or "", xlabel=r"$-\log_{10}(p)$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def corr_heatmap(df: pd.DataFrame, *, title: str | None = None) -> Figure:
    """Annotated correlation heatmap."""
    n = max(6, min(20, df.shape[1]))
    fig, ax = plt.subplots(figsize=(n, n * 0.8))
    sns.heatmap(df.corr(), cmap="RdBu_r", annot=True, fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set(title=title or "Correlation matrix")
    fig.tight_layout()
    return fig


def coef_bar(betas, feature_names, *, title: str | None = None) -> Figure:
    """Bar chart of model coefficients."""
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.6), 5))
    ax.bar([str(f) for f in feature_names], np.asarray(betas, float))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=title or "Coefficients", ylabel="Coefficient value")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    return fig


def roc_plot(fpr, tpr, auc, *, threshold_point=None, title: str | None = None) -> Figure:
    """ROC curve with the diagonal reference and an optional operating point."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color=ACCENT)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    if threshold_point is not None:
        ax.scatter(*threshold_point, color="red", zorder=5, label="Operating point")
    ax.set(title=title or "ROC curve", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend()
    fig.tight_layout()
    return fig


def confusion_plot(cm, labels, *, title: str | None = None) -> Figure:
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set(title=title or "Confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str, out_dir=Path("reports/figures")) -> Path:
    """Write a figure to ``out_dir/<name>.png`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
