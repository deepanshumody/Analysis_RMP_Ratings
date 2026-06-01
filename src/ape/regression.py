"""From-scratch linear models (OLS, ridge, lasso) and regression metrics.

Implemented with numpy to demonstrate the underlying math; validated against
scikit-learn in the test suite. The intercept is the first column (all ones)
and is never penalized.
"""

from __future__ import annotations

import numpy as np


def add_intercept(X) -> np.ndarray:
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict(X, beta) -> np.ndarray:
    return np.asarray(X, float) @ np.asarray(beta, float)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def fit_ols(X, y) -> np.ndarray:
    """Ordinary least squares via the normal equations (pseudo-inverse for stability)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def fit_ridge(X, y, alpha: float) -> np.ndarray:
    """Ridge regression; the intercept (column 0) is not penalized."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    p = X.shape[1]
    penalty = np.eye(p)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(X.T @ X + alpha * penalty) @ X.T @ y


def fit_lasso(X, y, alpha: float, max_iter: int = 10000, tol: float = 1e-7) -> np.ndarray:
    """Lasso via coordinate descent with soft-thresholding; intercept unpenalized."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    col_ss = (X ** 2).sum(axis=0)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            residual = y - X @ beta + X[:, j] * beta[j]
            rho = X[:, j] @ residual
            if j == 0:                      # intercept: no shrinkage
                beta[j] = rho / col_ss[j]
            else:
                beta[j] = np.sign(rho) * max(0.0, abs(rho) - alpha) / col_ss[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta
