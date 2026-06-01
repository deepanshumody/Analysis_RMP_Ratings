import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ape import classify


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logit = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_fit_returns_pipeline_with_scaler_inside():
    X, y = _data()
    model = classify.fit_pepper_model(X, y)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_youden_threshold_in_unit_interval():
    X, y = _data()
    model = classify.fit_pepper_model(X, y)
    score = model.predict_proba(X)[:, 1]
    thr = classify.youden_threshold(y, score)
    assert 0.0 <= thr <= 1.0


def test_evaluate_reports_auc():
    X, y = _data()
    model = classify.fit_pepper_model(X, y)
    out = classify.evaluate(model, X, y)
    assert 0.5 <= out["auc"] <= 1.0
    assert {"auc", "threshold", "confusion", "report"}.issubset(out)
