import numpy as np

from ape import regression as reg


def _data(seed=0, n=200, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.5, -2.0, 0.5])
    y = 3.0 + X @ beta + rng.normal(scale=0.1, size=n)
    return X, y


def test_ols_predictions_match_sklearn():
    from sklearn.linear_model import LinearRegression
    X, y = _data()
    Xd = reg.add_intercept(X)
    beta = reg.fit_ols(Xd, y)
    yhat = reg.predict(Xd, beta)
    sk = LinearRegression().fit(X, y).predict(X)
    assert np.allclose(yhat, sk, atol=1e-8)


def test_metrics_match_sklearn():
    from sklearn.metrics import mean_squared_error, r2_score
    X, y = _data()
    Xd = reg.add_intercept(X)
    yhat = reg.predict(Xd, reg.fit_ols(Xd, y))
    assert np.isclose(reg.rmse(y, yhat), np.sqrt(mean_squared_error(y, yhat)))
    assert np.isclose(reg.r2(y, yhat), r2_score(y, yhat))


def test_ridge_shrinks_coefficients():
    X, y = _data()
    Xd = reg.add_intercept(X)
    b0 = reg.fit_ridge(Xd, y, alpha=0.0)
    b_big = reg.fit_ridge(Xd, y, alpha=100.0)
    assert np.linalg.norm(b_big[1:]) < np.linalg.norm(b0[1:])


def test_ridge_alpha_zero_equals_ols():
    X, y = _data()
    Xd = reg.add_intercept(X)
    assert np.allclose(reg.fit_ridge(Xd, y, alpha=0.0), reg.fit_ols(Xd, y), atol=1e-6)


def test_lasso_alpha_zero_close_to_ols():
    X, y = _data()
    Xd = reg.add_intercept(X)
    b0 = reg.fit_lasso(Xd, y, alpha=0.0)
    assert np.allclose(b0, reg.fit_ols(Xd, y), atol=1e-2)


def test_lasso_zeros_irrelevant_feature():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x_noise = rng.normal(size=n)  # genuinely irrelevant predictor
    y = 2.0 + 3.0 * x1 + rng.normal(scale=0.1, size=n)
    Xd = reg.add_intercept(np.column_stack([x1, x_noise]))
    b = reg.fit_lasso(Xd, y, alpha=50.0)
    assert abs(b[2]) < 1e-6   # irrelevant feature driven to exactly zero
    assert abs(b[1]) > 0.1    # the real predictor survives
