"""Question-by-question analysis orchestrators (Q1-Q11).

Each ``qN()`` builds its analysis frame from :mod:`ape.data`, runs the
corrected analysis, and returns ``(results, figures)`` where ``results`` is a
JSON-friendly dict and ``figures`` maps names to matplotlib Figures.

Methodology corrections vs. the original capstone code:
- Q1 uses a one-sided ("greater") Mann-Whitney for the directional pro-male claim.
- Q4 applies Benjamini-Hochberg FDR across the 20 tags at alpha = 0.005.
- Q7-Q9 use leak-free k-fold CV (scaler fit inside each fold) + forward selection.
- Q10 scales inside a Pipeline (no train/test leakage), balances classes, and
  picks the decision threshold via Youden's J.
- Tags are normalized once, consistently, by each professor's number of ratings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import classify, config, data, effects, viz
from . import model_selection as ms
from . import stats_tests as st

N_BOOT = 2000  # bootstrap resamples for CIs (kept modest so the suite stays fast)


# --------------------------------------------------------------------------- #
# Shared frame builders
# --------------------------------------------------------------------------- #
def _numeric_ge10() -> pd.DataFrame:
    return data.filter_min_ratings(data.load_numeric())


def _gendered() -> pd.DataFrame:
    return data.gender_subset(_numeric_ge10())


def _split_by_gender(df: pd.DataFrame, column: str):
    male = df.loc[df["high_conf_male"] == 1, column].dropna().to_numpy()
    female = df.loc[df["high_conf_female"] == 1, column].dropna().to_numpy()
    return male, female


def _normalized_tag_frame() -> pd.DataFrame:
    """≥10-rating professors with tags normalized by number of ratings."""
    num = _numeric_ge10()
    tags = data.load_tags().loc[num.index]
    norm = data.normalize_tags(tags, denom=num["num_ratings"])
    return num.join(norm)


# --------------------------------------------------------------------------- #
# Q1 — pro-male gender bias in average rating (directional)
# --------------------------------------------------------------------------- #
def q1():
    df = _gendered()
    male, female = _split_by_gender(df, "avg_rating")
    mwu_greater = st.mann_whitney(male, female, alternative="greater")
    mwu_two = st.mann_whitney(male, female, alternative="two-sided")
    ks = st.ks_test(male, female)
    results = {
        "n_male": int(len(male)),
        "n_female": int(len(female)),
        "male_mean": float(np.mean(male)),
        "female_mean": float(np.mean(female)),
        "mwu_p_one_sided_greater": mwu_greater.pvalue,
        "mwu_p_two_sided": mwu_two.pvalue,
        "ks_p": ks.pvalue,
        "significant_pro_male": mwu_greater.significant,
    }
    figs = {"q1_rating_by_gender": viz.density_plot(
        male, female, "Male", "Female",
        title="Average professor rating by gender (>=10 ratings)")}
    return results, figs


# --------------------------------------------------------------------------- #
# Q2 — gender difference in spread (variance) of ratings
# --------------------------------------------------------------------------- #
def q2():
    df = _gendered()
    male, female = _split_by_gender(df, "avg_rating")
    lev = st.levene_test(male, female)
    results = {
        "male_var": float(np.var(male, ddof=1)),
        "female_var": float(np.var(female, ddof=1)),
        "levene_stat": lev.statistic,
        "levene_p": lev.pvalue,
        "significant": lev.significant,
    }
    figs = {"q2_spread_by_gender": viz.density_plot(
        male, female, "Male", "Female",
        title="Rating distribution spread by gender")}
    return results, figs


# --------------------------------------------------------------------------- #
# Q3 — effect sizes (mean shift + spread) with 95% CIs
# --------------------------------------------------------------------------- #
def q3():
    df = _gendered()
    male, female = _split_by_gender(df, "avg_rating")
    d, d_lo, d_hi = effects.cohen_d_ci(male, female, n_boot=N_BOOT)
    vr = effects.variance_ratio(male, female)
    vr_lo, vr_hi = effects.bootstrap_ci(effects.variance_ratio, male, female, n_boot=N_BOOT)
    # bootstrap distribution for the figure
    rng = np.random.default_rng(config.RANDOM_SEED)
    boot = np.array([
        effects.cohen_d(rng.choice(male, len(male), True), rng.choice(female, len(female), True))
        for _ in range(N_BOOT)
    ])
    results = {
        "cohen_d": d, "cohen_d_ci": [d_lo, d_hi],
        "variance_ratio": vr, "variance_ratio_ci": [vr_lo, vr_hi],
    }
    figs = {"q3_effect_size": viz.bootstrap_effect_plot(
        boot, d_lo, d_hi, title="Bootstrap distribution of Cohen's d (rating, M vs F)")}
    return results, figs


# --------------------------------------------------------------------------- #
# Q4 — gender difference across the 20 tags (FDR-corrected)
# --------------------------------------------------------------------------- #
def q4():
    df = data.gender_subset(_normalized_tag_frame())
    rows = []
    for tag in config.TAG_COLS:
        male = df.loc[df["high_conf_male"] == 1, tag].dropna()
        female = df.loc[df["high_conf_female"] == 1, tag].dropna()
        mwu = st.mann_whitney(male, female, alternative="two-sided")
        ks = st.ks_test(male, female)
        rows.append({"tag": tag, "mwu_p": mwu.pvalue, "ks_p": ks.pvalue})
    pdf = pd.DataFrame(rows)
    reject, p_adj = st.fdr_bh(pdf["mwu_p"].to_numpy(), alpha=config.ALPHA)
    pdf["mwu_p_fdr"] = p_adj
    pdf["significant_fdr"] = reject
    ordered = pdf.sort_values("mwu_p")
    results = {
        "n_significant_fdr": int(reject.sum()),
        "significant_tags": ordered.loc[ordered["significant_fdr"], "tag"].tolist(),
        "most_gendered": ordered["tag"].head(3).tolist(),
        "least_gendered": ordered["tag"].tail(3).tolist(),
    }
    labels = [config.TAG_LABELS[t] for t in pdf["tag"]]
    figs = {"q4_tag_significance": viz.tag_significance_bar(
        labels, pdf["mwu_p_fdr"].to_numpy(), alpha=config.ALPHA,
        title="Gender difference by tag (FDR-adjusted)")}
    return results, figs


# --------------------------------------------------------------------------- #
# Q5 — gender difference in average difficulty
# --------------------------------------------------------------------------- #
def q5():
    df = _gendered()
    male, female = _split_by_gender(df, "avg_difficulty")
    mwu = st.mann_whitney(male, female, alternative="two-sided")
    ks = st.ks_test(male, female)
    results = {
        "male_mean": float(np.mean(male)), "female_mean": float(np.mean(female)),
        "mwu_p": mwu.pvalue, "ks_p": ks.pvalue,
        "significant": mwu.significant or ks.significant,
    }
    figs = {"q5_difficulty_by_gender": viz.density_plot(
        male, female, "Male", "Female", title="Average difficulty by gender")}
    return results, figs


# --------------------------------------------------------------------------- #
# Q6 — effect size of the difficulty difference (95% CI)
# --------------------------------------------------------------------------- #
def q6():
    df = _gendered()
    male, female = _split_by_gender(df, "avg_difficulty")
    d, d_lo, d_hi = effects.cohen_d_ci(male, female, n_boot=N_BOOT)
    rng = np.random.default_rng(config.RANDOM_SEED)
    boot = np.array([
        effects.cohen_d(rng.choice(male, len(male), True), rng.choice(female, len(female), True))
        for _ in range(N_BOOT)
    ])
    results = {"cohen_d": d, "cohen_d_ci": [d_lo, d_hi]}
    figs = {"q6_difficulty_effect": viz.bootstrap_effect_plot(
        boot, d_lo, d_hi, title="Bootstrap Cohen's d (difficulty, M vs F)")}
    return results, figs


# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #
def _final_coefficients(X: pd.DataFrame, y: np.ndarray):
    """Standardized OLS coefficients on the full design (for interpretation)."""
    from sklearn.preprocessing import StandardScaler

    from . import regression as reg
    Xs = StandardScaler().fit_transform(X.to_numpy())
    beta = reg.fit_ols(reg.add_intercept(Xs), y)
    return dict(zip(X.columns, beta[1:], strict=True))


def _regression_question(X: pd.DataFrame, y: np.ndarray, max_features: int | None,
                         label_map: dict | None = None):
    selected, history = ms.forward_selection(X, y, max_features=max_features)
    best = min(history, key=lambda h: h["rmse"])
    coefs = _final_coefficients(X[best["features"]], y)
    top = max(coefs, key=lambda k: abs(coefs[k]))
    results = {
        "best_features": best["features"],
        "best_rmse": float(best["rmse"]),
        "best_r2": float(best["r2"]),
        "top_predictor": top,
        "top_predictor_beta": float(coefs[top]),
        "forward_path": [(h["features"][-1], round(h["r2"], 3)) for h in history],
    }
    names = [(label_map or {}).get(k, k) for k in coefs]
    fig = viz.coef_bar(list(coefs.values()), names,
                       title="Standardized coefficients (selected model)")
    return results, coefs, fig


# --------------------------------------------------------------------------- #
# Q7 — predict average rating from numerical predictors
# --------------------------------------------------------------------------- #
def q7():
    df = data.gender_subset(_numeric_ge10()).copy()
    df["prop_online"] = df["num_online"] / df["num_ratings"]
    df = df.dropna(subset=["prop_retake"])
    features = ["avg_difficulty", "num_ratings", "received_pepper",
                "high_conf_female", "prop_online", "prop_retake"]
    X, y = df[features], df["avg_rating"].to_numpy()
    results, _, fig = _regression_question(X, y, max_features=None)
    return results, {"q7_coefficients": fig}


# --------------------------------------------------------------------------- #
# Q8 — predict average rating from tags
# --------------------------------------------------------------------------- #
def q8():
    df = _normalized_tag_frame().dropna(subset=["avg_rating"])
    X, y = df[config.TAG_COLS], df["avg_rating"].to_numpy()
    results, _, fig = _regression_question(X, y, max_features=8, label_map=config.TAG_LABELS)
    return results, {"q8_coefficients": fig}


# --------------------------------------------------------------------------- #
# Q9 — predict average difficulty from tags
# --------------------------------------------------------------------------- #
def q9():
    df = _normalized_tag_frame().dropna(subset=["avg_difficulty"])
    X, y = df[config.TAG_COLS], df["avg_difficulty"].to_numpy()
    results, _, fig = _regression_question(X, y, max_features=8, label_map=config.TAG_LABELS)
    return results, {"q9_coefficients": fig}


# --------------------------------------------------------------------------- #
# Q10 — predict "received a pepper" from all features
# --------------------------------------------------------------------------- #
def q10():
    df = data.gender_subset(_normalized_tag_frame()).copy()
    df["prop_online"] = df["num_online"] / df["num_ratings"]
    df = df.dropna()
    num_features = ["avg_rating", "avg_difficulty", "num_ratings", "prop_retake", "prop_online"]
    features = num_features + config.TAG_COLS
    X = df[features].to_numpy()
    y = df["received_pepper"].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)
    model = classify.fit_pepper_model(X_tr, y_tr)
    out = classify.evaluate(model, X_te, y_te)
    report = out["report"]
    results = {
        "n": int(len(df)),
        "pepper_rate": float(y.mean()),
        "auc": out["auc"],
        "threshold": out["threshold"],
        "f1_class0": float(report["0"]["f1-score"]),
        "f1_class1": float(report["1"]["f1-score"]),
        "confusion": out["confusion"].tolist(),
    }
    fpr, tpr, _ = out["roc"]
    figs = {
        "q10_roc": viz.roc_plot(fpr, tpr, out["auc"], title="Pepper classifier ROC"),
        "q10_confusion": viz.confusion_plot(out["confusion"], ["No pepper", "Pepper"]),
    }
    return results, figs


# --------------------------------------------------------------------------- #
# Q11 — bonus: do NY and NJ professors differ in average rating?
# --------------------------------------------------------------------------- #
def q11():
    num = _numeric_ge10()
    qual = data.load_qual().loc[num.index]
    df = num.join(qual["state"])
    ny = df.loc[df["state"] == "NY", "avg_rating"].dropna().to_numpy()
    nj = df.loc[df["state"] == "NJ", "avg_rating"].dropna().to_numpy()
    mwu = st.mann_whitney(ny, nj, alternative="two-sided")
    ks = st.ks_test(ny, nj)
    results = {
        "n_ny": int(len(ny)), "n_nj": int(len(nj)),
        "ny_mean": float(np.mean(ny)), "nj_mean": float(np.mean(nj)),
        "mwu_p": mwu.pvalue, "ks_p": ks.pvalue,
        "significant": mwu.significant or ks.significant,
    }
    figs = {"q11_ny_vs_nj": viz.density_plot(
        ny, nj, "New York", "New Jersey", title="Average rating: NY vs NJ")}
    return results, figs


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_all() -> dict:
    out = {}
    for i in range(1, 12):
        results, _ = globals()[f"q{i}"]()
        out[f"q{i}"] = results
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2, default=str))
