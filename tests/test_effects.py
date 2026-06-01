import numpy as np

from ape import effects


def test_cohen_d_known_value():
    a = np.array([2.0, 4.0, 6.0, 8.0])
    b = np.array([1.0, 3.0, 5.0, 7.0])
    d = effects.cohen_d(a, b)
    assert d > 0
    assert np.isclose(d, 1.0 / np.std(a, ddof=1))


def test_cohen_d_sign_flips():
    a, b = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
    assert effects.cohen_d(a, b) == -effects.cohen_d(b, a)


def test_cohen_d_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    a = rng.normal(0.3, 1, 500)
    b = rng.normal(0.0, 1, 500)
    d, lo, hi = effects.cohen_d_ci(a, b, n_boot=500, seed=1)
    assert lo < d < hi


def test_variance_ratio():
    a = np.array([1.0, 2, 3, 4])
    b = np.array([1.0, 1, 1, 1.001])
    assert effects.variance_ratio(a, b) > 1


def test_rank_biserial_range():
    assert -1.0 <= effects.rank_biserial(u=10, n1=5, n2=5) <= 1.0
