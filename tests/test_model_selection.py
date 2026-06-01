import numpy as np
import pandas as pd

from ape import model_selection as ms


def _frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = 1.0 + 2 * X["a"] - 1.5 * X["b"] + rng.normal(scale=0.2, size=n)
    return X, y.to_numpy()


def test_cross_val_scores_returns_k_folds():
    X, y = _frame()
    res = ms.cross_val_scores(X.to_numpy(), y, k=5, seed=1)
    assert len(res["rmse"]) == 5 and len(res["r2"]) == 5
    assert np.mean(res["r2"]) > 0.8


def test_forward_selection_picks_informative_first():
    X, y = _frame()
    selected, history = ms.forward_selection(X, y, k=5, seed=1)
    assert selected[0] in {"a", "b"}
    assert len(history) == X.shape[1]
    rmses = [h["rmse"] for h in history]
    assert rmses[0] >= rmses[-1]
