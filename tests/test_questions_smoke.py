import pytest
from matplotlib.figure import Figure

from ape import questions


@pytest.mark.parametrize("fn_name", [f"q{i}" for i in range(1, 12)])
def test_question_runs_and_returns_results_and_figures(fn_name):
    fn = getattr(questions, fn_name)
    results, figures = fn()
    assert isinstance(results, dict) and len(results) >= 1
    assert all(isinstance(f, Figure) for f in figures.values())


def test_q7_rating_regression_reasonable():
    results, _ = questions.q7()
    assert 0.5 <= results["best_r2"] <= 0.95


def test_q10_auc_reasonable():
    results, _ = questions.q10()
    assert 0.7 <= results["auc"] <= 0.95
