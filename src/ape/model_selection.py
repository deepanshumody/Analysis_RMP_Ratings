"""Leak-free k-fold cross-validation and forward feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import config
from . import regression as reg


def cross_val_scores(X, y, k: int = 5, seed: int = config.RANDOM_SEED) -> dict:
    """K-fold CV for OLS with per-fold standardization (fit on train only)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rmses, r2s = [], []
    for tr, va in kf.split(X):
        scaler = StandardScaler().fit(X[tr])            # fit on TRAIN only
        Xtr = reg.add_intercept(scaler.transform(X[tr]))
        Xva = reg.add_intercept(scaler.transform(X[va]))
        beta = reg.fit_ols(Xtr, y[tr])
        yhat = reg.predict(Xva, beta)
        rmses.append(reg.rmse(y[va], yhat))
        r2s.append(reg.r2(y[va], yhat))
    return {"rmse": rmses, "r2": r2s}


def forward_selection(X: pd.DataFrame, y, k: int = 5, seed: int = config.RANDOM_SEED,
                      max_features: int | None = None):
    """Greedy forward selection by mean CV RMSE. Returns (selected_cols, history).

    Stops after ``max_features`` have been chosen (default: use all columns).
    """
    y = np.asarray(y, float)
    remaining = list(X.columns)
    selected: list[str] = []
    history = []
    limit = max_features or len(remaining)
    while remaining and len(selected) < limit:
        best_feat, best_rmse, best_r2 = None, np.inf, None
        for feat in remaining:
            cols = selected + [feat]
            res = cross_val_scores(X[cols].to_numpy(), y, k=k, seed=seed)
            mean_rmse = float(np.mean(res["rmse"]))
            if mean_rmse < best_rmse:
                best_feat, best_rmse, best_r2 = feat, mean_rmse, float(np.mean(res["r2"]))
        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append({"features": selected.copy(), "rmse": best_rmse, "r2": best_r2})
    return selected, history
