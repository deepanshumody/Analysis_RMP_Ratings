import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ape import viz


def test_density_plot_returns_figure_and_saves(tmp_path):
    a = pd.Series(np.random.default_rng(0).normal(size=100))
    b = pd.Series(np.random.default_rng(1).normal(size=100))
    fig = viz.density_plot(a, b, "Group A", "Group B", title="t")
    assert isinstance(fig, Figure)
    out = viz.save_figure(fig, "demo", out_dir=tmp_path)
    assert out.exists()


def test_roc_and_confusion_return_figures():
    fpr = np.array([0.0, 0.2, 1.0])
    tpr = np.array([0.0, 0.7, 1.0])
    assert isinstance(viz.roc_plot(fpr, tpr, 0.82), Figure)
    assert isinstance(viz.confusion_plot([[10, 2], [3, 8]], ["No", "Yes"]), Figure)


def test_tag_significance_bar_returns_figure():
    labels = ["Hilarious", "Caring", "Pop quizzes!"]
    pvals = [1e-30, 1e-5, 0.7]
    assert isinstance(viz.tag_significance_bar(labels, pvals, alpha=0.005), Figure)
