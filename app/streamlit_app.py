"""Interactive dashboard for the APE analysis.

Run with:  streamlit run app/streamlit_app.py

This is an interactive companion to the static narrative site. It reuses the
same `ape` package, so the numbers and figures match the report exactly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from ape import config, data, effects, viz
from ape import stats_tests as st_tests

st.set_page_config(page_title="APE — Professor Effectiveness", layout="wide")


@st.cache_data
def load():
    return data.load_numeric()


df_all = load()

st.title("Assessing Professor Effectiveness")
st.caption(
    "Interactive companion to the analysis of 89,893 RateMyProfessor records "
    "(NYU DS-GA 1001 capstone, group CAP 85). Built on the `ape` package."
)

# ---- Sidebar controls ----------------------------------------------------- #
with st.sidebar:
    st.header("Filters")
    min_ratings = st.slider("Minimum number of ratings", 1, 50, config.MIN_RATINGS)
    metric = st.selectbox(
        "Outcome",
        options=["avg_rating", "avg_difficulty"],
        format_func=lambda c: {"avg_rating": "Average rating",
                               "avg_difficulty": "Average difficulty"}[c],
    )
    pepper = st.radio("Pepper status", ["All", "Pepper only", "No pepper"], horizontal=False)

# ---- Build the working frame ---------------------------------------------- #
work = data.gender_subset(data.filter_min_ratings(df_all, k=min_ratings))
if pepper == "Pepper only":
    work = work[work["received_pepper"] == 1]
elif pepper == "No pepper":
    work = work[work["received_pepper"] == 0]

male = work.loc[work["high_conf_male"] == 1, metric].dropna()
female = work.loc[work["high_conf_female"] == 1, metric].dropna()

label = {"avg_rating": "Average rating", "avg_difficulty": "Average difficulty"}[metric]

col1, col2, col3 = st.columns(3)
col1.metric("Professors (filtered)", f"{len(work):,}")
col2.metric("Male", f"{len(male):,}")
col3.metric("Female", f"{len(female):,}")

tab1, tab2 = st.tabs(["Distribution & tests", "Effect size"])

with tab1:
    fig = viz.density_plot(male, female, "Male", "Female", title=f"{label} by gender")
    st.pyplot(fig)
    plt.close(fig)

    mwu = st_tests.mann_whitney(male, female, alternative="two-sided")
    ks = st_tests.ks_test(male, female)
    c1, c2 = st.columns(2)
    c1.metric("Mann-Whitney U p-value", f"{mwu.pvalue:.2e}",
              "significant" if mwu.significant else "n.s. (α=0.005)")
    c2.metric("KS test p-value", f"{ks.pvalue:.2e}",
              "significant" if ks.significant else "n.s. (α=0.005)")

with tab2:
    if len(male) > 1 and len(female) > 1:
        d, lo, hi = effects.cohen_d_ci(male.to_numpy(), female.to_numpy(), n_boot=1000)
        st.metric("Cohen's d (male − female)", f"{d:.3f}", f"95% CI [{lo:.3f}, {hi:.3f}]")
        st.caption(
            "A positive d means male professors score higher. |d| < 0.2 is a small effect. "
            "The CI is a 1,000-sample bootstrap."
        )
    else:
        st.info("Not enough data in one group for an effect-size estimate. Loosen the filters.")

st.divider()
st.caption("Source & full narrative: github.com/deepanshumody/Analysis_RMP_Ratings")
