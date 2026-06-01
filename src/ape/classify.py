"""Leak-free logistic-regression classifier for the 'received a pepper' target."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def fit_pepper_model(X_train, y_train, class_weight: str | None = "balanced") -> Pipeline:
    """Standardize + logistic regression in a single Pipeline (scaler fit on train only)."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000,
                                   random_state=config.RANDOM_SEED)),
    ])
    model.fit(X_train, y_train)
    return model


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """AUROC + threshold-consistent confusion matrix and classification report."""
    score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, score)
    thr = youden_threshold(y_test, score)
    y_pred = (score >= thr).astype(int)
    return {
        "auc": float(auc),
        "threshold": thr,
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "roc": roc_curve(y_test, score),
    }
