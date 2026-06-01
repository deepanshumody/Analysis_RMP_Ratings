"""Plotting helpers with a cohesive Swiss/minimal style.

Every function builds and returns a ``matplotlib.figure.Figure``; nothing is
shown or written to disk implicitly. ``save_figure`` is the only function that
touches the filesystem. Importing this module applies a consistent house style
(black + cobalt on white, hairline grids, restrained type) so every figure
across the site, notebook, and app looks like one system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# --- House palette (Swiss: ink + cobalt on white) ---
INK = "#0f0f0f"
COBALT = "#1b3bdb"        # the single accent
GREY = "#bdbdbd"          # de-emphasized / secondary
GRIDC = "#ededed"
LINE = "#111111"

# Backwards-compatible names used elsewhere
MALE_COLOR = INK
FEMALE_COLOR = COBALT
ACCENT = COBALT


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Archivo", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDC,
        "grid.linewidth": 1.0,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


_apply_style()


def _hairline(ax, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=1.0)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def density_plot(s1, s2, label1, label2, *, bins: int = 30, title: str | None = None) -> Figure:
    """Overlaid normalized histograms (KDE) for two groups."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(s1, bins=bins, kde=True, color=INK, label=label1, stat="density",
                 ax=ax, alpha=0.32, edgecolor="white", linewidth=0.4)
    sns.histplot(s2, bins=bins, kde=True, color=COBALT, label=label2, stat="density",
                 ax=ax, alpha=0.32, edgecolor="white", linewidth=0.4)
    ax.set(title=title or "", ylabel="Density", xlabel="")
    _hairline(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def ci_plot(series, *, title: str | None = None) -> Figure:
    """Histogram with mean and 95% CI of the mean marked."""
    series = pd.Series(series).dropna()
    mean = series.mean()
    half = 1.96 * series.std() / np.sqrt(len(series))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(series, bins=30, kde=True, color=INK, ax=ax, alpha=0.28,
                 edgecolor="white", linewidth=0.4)
    ax.axvline(mean - half, color=COBALT, linestyle="--", label=f"Lower CI: {mean - half:.2f}")
    ax.axvline(mean, color=INK, label=f"Mean: {mean:.2f}")
    ax.axvline(mean + half, color=COBALT, linestyle="--", label=f"Upper CI: {mean + half:.2f}")
    ax.set(title=title or "", ylabel="Density", xlabel="")
    _hairline(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def bootstrap_effect_plot(values, lo, hi, *, title: str | None = None) -> Figure:
    """Bootstrap distribution of an effect size with its CI bounds marked."""
    values = np.asarray(values, float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(values, bins=30, kde=True, color=COBALT, ax=ax, alpha=0.28,
                 edgecolor="white", linewidth=0.4)
    ax.axvline(lo, color=INK, linestyle="--", label=f"Lower: {lo:.3f}")
    ax.axvline(values.mean(), color=COBALT, label=f"Mean: {values.mean():.3f}")
    ax.axvline(hi, color=INK, linestyle="--", label=f"Upper: {hi:.3f}")
    ax.axvline(0, color=GREY, linewidth=1.0)
    ax.set(title=title or "", xlabel="Effect size", ylabel="Frequency")
    _hairline(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def pval_scatter(df_pvals: pd.DataFrame, *, title: str | None = None) -> Figure:
    """Scatter of per-tag p-values (Mann-Whitney vs KS), labelled by tag."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_pvals["mwu_p"], df_pvals["ks_p"], color=COBALT, s=28, zorder=3)
    for _, row in df_pvals.iterrows():
        ax.text(row["mwu_p"], row["ks_p"], row["tag"], fontsize=7, rotation=30, color=INK)
    ax.set(title=title or "P-values by tag", xlabel="Mann-Whitney U p-value",
           ylabel="KS test p-value")
    _hairline(ax, grid_axis="both")
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
    colors = [COBALT if v >= thr else GREY for v in neg]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, neg, color=colors, height=0.7)
    ax.axvline(thr, color=INK, linestyle="--", linewidth=1.3, label=f"α = {alpha} (FDR)")
    ax.set(title=title or "", xlabel=r"$-\log_{10}(p)$", ylabel="")
    _hairline(ax, grid_axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def corr_heatmap(df: pd.DataFrame, *, title: str | None = None) -> Figure:
    """Annotated correlation heatmap."""
    n = max(6, min(20, df.shape[1]))
    fig, ax = plt.subplots(figsize=(n, n * 0.8))
    sns.heatmap(df.corr(), cmap="RdBu_r", annot=True, fmt=".2f", ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.6})
    ax.set(title=title or "Correlation matrix")
    fig.tight_layout()
    return fig


def coef_bar(betas, feature_names, *, title: str | None = None) -> Figure:
    """Bar chart of model coefficients, coloured by sign (cobalt +, ink −)."""
    betas = np.asarray(betas, float)
    colors = [COBALT if b >= 0 else INK for b in betas]
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.62), 4.8))
    ax.bar([str(f) for f in feature_names], betas, color=colors, width=0.64)
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.set(title=title or "Coefficients", ylabel="Standardized β")
    _hairline(ax, grid_axis="y")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    return fig


def roc_plot(fpr, tpr, auc, *, threshold_point=None, title: str | None = None) -> Figure:
    """ROC curve with the diagonal reference and an optional operating point."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color=COBALT, linewidth=2.4)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1.0)
    if threshold_point is not None:
        ax.scatter(*threshold_point, color=INK, zorder=5, label="Operating point")
    ax.set(title=title or "ROC curve", xlabel="False positive rate",
           ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    _hairline(ax, grid_axis="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def confusion_plot(cm, labels, *, title: str | None = None) -> Figure:
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False, linewidths=2, linecolor="white",
                annot_kws={"size": 13})
    ax.set(title=title or "Confusion matrix", xlabel="Predicted", ylabel="Actual")
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str, out_dir=Path("reports/figures")) -> Path:
    """Write a figure to ``out_dir/<name>.png`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path
