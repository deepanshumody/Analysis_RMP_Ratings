import numpy as np

from ape import stats_tests as st


def test_mann_whitney_matches_scipy():
    from scipy import stats
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.array([2.0, 3, 4, 5, 6])
    r = st.mann_whitney(a, b, alternative="two-sided")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert np.isclose(r.statistic, u) and np.isclose(r.pvalue, p)


def test_mann_whitney_directional_smaller_pvalue():
    rng = np.random.default_rng(0)
    male = rng.normal(0.2, 1, 300)
    female = rng.normal(0.0, 1, 300)
    two = st.mann_whitney(male, female, alternative="two-sided").pvalue
    one = st.mann_whitney(male, female, alternative="greater").pvalue
    assert one < two


def test_result_significance_uses_alpha():
    r = st.ks_test(np.arange(50.0), np.arange(50.0) + 5)
    assert r.significant == (r.pvalue < 0.005)


def test_fdr_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.5])
    reject, p_adj = st.fdr_bh(p, alpha=0.005)
    exp_reject, exp_p, *_ = multipletests(p, alpha=0.005, method="fdr_bh")
    assert np.array_equal(reject, exp_reject)
    assert np.allclose(p_adj, exp_p)
