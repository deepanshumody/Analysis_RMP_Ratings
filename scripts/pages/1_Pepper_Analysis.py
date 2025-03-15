############################################################
# streamlit_q1to6.py
# A Comprehensive Interactive App for Q1–Q6
# (Assessing Professor Effectiveness Project)
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, levene
import warnings

# --------------------------------------------------------------------
# 1) HELPER / UTILITY FUNCTIONS
# --------------------------------------------------------------------
def load_data(num_path="./data/rmpCapstoneNum.csv", tags_path="./data/rmpCapstoneTags.csv"):
    """
    Loads numeric data (rmpCapstoneNum.csv) and tag data (rmpCapstoneTags.csv),
    filters to rows with >=10 ratings and exactly one gender=1.

    Returns:
      df_10plus: A numeric-only DataFrame with columns:
         - 'AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings',
           'Received a pepper', 'Proportion of students that said they would take the class again',
           'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale'
      df_tags_10plus: The same subset but includes 20 additional tag columns as proportions.
    """
    # 1) Numeric data (rmpCapstoneNum)
    df_num = pd.read_csv(num_path, header=0)
    df_num.columns = [
        'AverageProfessorRating', 'Average Difficulty',
        'NumberOfRatings', 'Received a pepper',
        'Proportion of students that said they would take the class again',
        'Number of ratings coming from online classes',
        'HighConfMale', 'HighConfFemale'
    ]
    # Filter for >=10 ratings
    df_10plus = df_num[df_num['NumberOfRatings'] >= 10].copy()
    # Keep exactly one gender=1
    df_10plus = df_10plus[~(
        ((df_10plus['HighConfMale'] == 1) & (df_10plus['HighConfFemale'] == 1)) |
        ((df_10plus['HighConfMale'] == 0) & (df_10plus['HighConfFemale'] == 0))
    )].copy()

    # 2) Tag data (rmpCapstoneTags)
    df_tags = pd.read_csv(tags_path, header=None)
    df_tags.columns = [
        "Tough grader", "Good feedback", "Respected", "Lots to read",
        "Participation matters", "Dont skip class", "Lots of homework",
        "Inspirational", "Pop quizzes!", "Accessible", "So many papers",
        "Clear grading", "Hilarious", "Test heavy", "Graded by few things",
        "Amazing lectures", "Caring", "Extra credit", "Group projects",
        "Lecture heavy"
    ]

    # Merge tags side-by-side with the original df_num
    df_tags_merged = pd.concat([df_num, df_tags], axis=1)

    # Filter the same way (≥10 ratings, exactly one gender=1)
    df_tags_10plus = df_tags_merged[df_tags_merged['NumberOfRatings'] >= 10].copy()
    df_tags_10plus = df_tags_10plus[~(
        ((df_tags_10plus['HighConfMale'] == 1) & (df_tags_10plus['HighConfFemale'] == 1)) |
        ((df_tags_10plus['HighConfMale'] == 0) & (df_tags_10plus['HighConfFemale'] == 0))
    )].copy()

    # Convert tag columns to proportions
    # Tag columns start at index 8+ (after the 8 numeric columns)
    tag_cols = df_tags_10plus.columns[8:]
    for col in tag_cols:
        df_tags_10plus[col] = df_tags_10plus[col].astype(float) / df_tags_10plus['NumberOfRatings']

    return df_10plus, df_tags_10plus

def distribution_plot(df_male, df_female, column, nbins=30):
    """
    Plots a side-by-side distribution (histogram + KDE) of df_male[column] vs df_female[column].
    """
    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(df_male[column].dropna(), bins=nbins, kde=True,
                 label="Male", stat='density', color='blue', ax=ax)
    sns.histplot(df_female[column].dropna(), bins=nbins, kde=True,
                 label="Female", stat='density', color='red', ax=ax)
    ax.set_title(f"Distribution of {column} by Gender (≥10 Ratings, HighConf)")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

def ks_mw_test(df_male, df_female, column, alpha=0.005):
    """
    Performs Kolmogorov-Smirnov (KS) & Mann-Whitney U (MW) tests on df_male[column] vs df_female[column].
    Returns (ks_stat, ks_p, ks_result_str), (mw_stat, mw_p, mw_result_str).
    """
    arr_m = df_male[column].dropna()
    arr_f = df_female[column].dropna()

    ks_stat, ks_p = ks_2samp(arr_m, arr_f)
    if ks_p < alpha:
        ks_result = f"Distributions differ significantly (p < {alpha})."
    else:
        ks_result = f"No distribution difference (p ≥ {alpha})."

    mw_stat, mw_p = mannwhitneyu(arr_m, arr_f)
    if mw_p < alpha:
        mw_result = f"Medians differ significantly (p < {alpha})."
    else:
        mw_result = f"No median difference (p ≥ {alpha})."

    return (ks_stat, ks_p, ks_result), (mw_stat, mw_p, mw_result)

def levenes_test(df_male, df_female, column, alpha=0.005):
    """
    Performs Levene's test for variance difference on df_male[column] vs df_female[column].
    Returns (stat, p_val, interpretation_str).
    """
    arr_m = df_male[column].dropna()
    arr_f = df_female[column].dropna()
    stat, p_val = levene(arr_m, arr_f)
    if p_val < alpha:
        interpretation = f"Variances differ significantly (p < {alpha})."
    else:
        interpretation = f"No variance difference (p ≥ {alpha})."
    return stat, p_val, interpretation

def cohen_d(df_male, df_female, column):
    """
    Computes Cohen's d for df_male[column] vs df_female[column].
    """
    arr_m = df_male[column].dropna()
    arr_f = df_female[column].dropna()
    if len(arr_m) < 2 or len(arr_f) < 2:
        return np.nan

    mean_diff = arr_m.mean() - arr_f.mean()
    pooled_var = (arr_m.var(ddof=1) + arr_f.var(ddof=1)) / 2
    if pooled_var < 1e-12:
        return 0.0
    return mean_diff / np.sqrt(pooled_var)

def bootstrap_cohen_d(df_male, df_female, column, n_boot=500, alpha=0.005):
    """
    Bootstraps the Cohen's d statistic for M vs F on the given column, returning (lower, upper, mean).
    """
    arr_m = df_male[column].dropna().values
    arr_f = df_female[column].dropna().values
    if len(arr_m) < 2 or len(arr_f) < 2:
        return (np.nan, np.nan, np.nan)

    n1, n2 = len(arr_m), len(arr_f)
    boot_vals = []
    for _ in range(n_boot):
        s1 = np.random.choice(arr_m, size=n1, replace=True)
        s2 = np.random.choice(arr_f, size=n2, replace=True)
        md = s1.mean() - s2.mean()
        pooled_var = (s1.var(ddof=1) + s2.var(ddof=1)) / 2
        if pooled_var < 1e-12:
            d_val = 0.0
        else:
            d_val = md / np.sqrt(pooled_var)
        boot_vals.append(d_val)

    boot_vals = np.array(boot_vals)
    lower = np.percentile(boot_vals, 100*(alpha/2))
    upper = np.percentile(boot_vals, 100*(1 - alpha/2))
    mean_ = boot_vals.mean()
    return (lower, upper, mean_)

def distribution_plot_tag(df_male, df_female, tag_column, nbins=30):
    """
    Plots distribution for a selected tag (already normalized) for M vs F.
    """
    arr_m = df_male[tag_column].dropna()
    arr_f = df_female[tag_column].dropna()

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(arr_m, bins=nbins, kde=True, label='Male', color='blue', ax=ax, stat='density')
    sns.histplot(arr_f, bins=nbins, kde=True, label='Female', color='red', ax=ax, stat='density')
    ax.set_title(f"Tag: {tag_column} (Proportion) – M vs F")
    ax.set_xlabel("Tag Proportion")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


# --------------------------------------------------------------------
# 2) STREAMLIT APP MAIN
# --------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    st.title("Interactive Analysis: Gender Differences, Variance, Effect Size, Tags, & Difficulty")

    # Load data once
    df_10plus, df_tags_10plus = load_data(
        num_path="./data/rmpCapstoneNum.csv",
        tags_path="./data/rmpCapstoneTags.csv"
    )

    # Split data by gender
    df_male   = df_10plus[df_10plus['HighConfMale'] == 1]
    df_female = df_10plus[df_10plus['HighConfFemale'] == 1]

    df_tags_male   = df_tags_10plus[df_tags_10plus['HighConfMale'] == 1]
    df_tags_female = df_tags_10plus[df_tags_10plus['HighConfFemale'] == 1]

    # Let user choose which question to explore
    question_choice = st.sidebar.radio(
        "Which question would you like to explore?",
        ("Q1: Compare Average Professor Rating by Gender", "Q2: Variance Differences by Gender (Levene's Test)", "Q3: Effect Size (Cohen's d) & CI for AverageProfessorRating", "Q4: Tag Analysis (Proportions) by Gender", "Q5: Compare Average Difficulty by Gender", "Q6: Effect Size for Average Difficulty by Pepper Subgroups"),
        index=0
    )

    # ========================= Q1 ==========================
    if question_choice == "Q1: Compare Average Professor Rating by Gender":
        st.header("Q1: Compare Average Professor Rating by Gender")
        st.markdown("""
        **Question 1** tests if M vs F differ in their **AverageProfessorRating**.  
        - We use Kolmogorov-Smirnov (KS) for distribution differences.  
        - We use Mann-Whitney U (MW) for median differences.  
        You can also filter by Pepper or #Ratings≥19.
        """)

        # Subset controls
        subset_choice = st.selectbox(
            "Choose a Subset of Data",
            ["All (≥10 Ratings)", "Pepper=Yes", "Pepper=No", "≥19 Ratings"]
        )

        # Filter data
        df_m = df_male.copy()
        df_f = df_female.copy()
        if subset_choice == "Pepper=Yes":
            df_m = df_m[df_m['Received a pepper'] == 1]
            df_f = df_f[df_f['Received a pepper'] == 1]
        elif subset_choice == "Pepper=No":
            df_m = df_m[df_m['Received a pepper'] == 0]
            df_f = df_f[df_f['Received a pepper'] == 0]
        elif subset_choice == "≥19 Ratings":
            df_m = df_m[df_m['NumberOfRatings'] >= 19]
            df_f = df_f[df_f['NumberOfRatings'] >= 19]

        st.subheader("Distribution Plot: AverageProfessorRating")
        distribution_plot(df_m, df_f, "AverageProfessorRating")

        st.subheader("KS & MW Tests")
        (ks_stat, ks_p, ks_str), (mw_stat, mw_p, mw_str) = ks_mw_test(df_m, df_f, "AverageProfessorRating")
        st.write(f"**KS**: stat={ks_stat:.4f}, p={ks_p:.3g} → {ks_str}")
        st.write(f"**MW**: stat={mw_stat:.4f}, p={mw_p:.3g} → {mw_str}")

    # ========================= Q2 ==========================
    elif question_choice == "Q2: Variance Differences by Gender (Levene's Test)":
        st.header("Q2: Variance Differences by Gender (Levene's Test)")
        st.markdown("""
        **Question 2** checks if the variance (spread) of M vs F differs for a chosen numeric column.  
        - We use Levene's Test to compare the variances.  
        - You can pick a numeric column below.
        """)

        numeric_columns = ["AverageProfessorRating", "Average Difficulty", "NumberOfRatings"]
        col_select = st.selectbox("Select a Numeric Column", numeric_columns)
        distribution_plot(df_male, df_female, col_select)

        st.write("### Levene's Test for Variance Difference")
        if st.button("Run Levene's Test"):
            stat, p_val, interp = levenes_test(df_male, df_female, col_select)
            st.write(f"Stat={stat:.4f}, p={p_val:.3g}, {interp}")

    # ========================= Q3 ==========================
    elif question_choice == "Q3: Effect Size (Cohen's d) & CI for AverageProfessorRating":
        st.header("Q3: Effect Size (Cohen's d) & CI for AverageProfessorRating")
        st.markdown("""
        **Question 3** focuses on the effect size (Cohen's d) for M vs F differences in 
        **AverageProfessorRating**.  
        - We optionally filter by Pepper or #Ratings≥19.  
        - Then we compute a bootstrap confidence interval for d.
        """)

        # Subset choice
        subset_choice = st.selectbox(
            "Subset for M vs F (Effect Size on AverageProfessorRating)",
            ["All (≥10 Ratings)", "Pepper=Yes", "Pepper=No", "≥19 Ratings"]
        )

        df_m = df_male.copy()
        df_f = df_female.copy()
        if subset_choice == "Pepper=Yes":
            df_m = df_m[df_m['Received a pepper'] == 1]
            df_f = df_f[df_f['Received a pepper'] == 1]
        elif subset_choice == "Pepper=No":
            df_m = df_m[df_m['Received a pepper'] == 0]
            df_f = df_f[df_f['Received a pepper'] == 0]
        elif subset_choice == "≥19 Ratings":
            df_m = df_m[df_m['NumberOfRatings'] >= 19]
            df_f = df_f[df_f['NumberOfRatings'] >= 19]

        st.subheader("Distribution Plot: AverageProfessorRating")
        distribution_plot(df_m, df_f, "AverageProfessorRating")

        # Cohen's d
        d_val = cohen_d(df_m, df_f, "AverageProfessorRating")
        st.write(f"**Cohen's d** for {subset_choice}: {d_val:.3f}" if not np.isnan(d_val) else "N/A")

        st.write("### Bootstrap Confidence Interval")
        if st.button("Compute Bootstrap CI"):
            lb, ub, mean_ = bootstrap_cohen_d(df_m, df_f, "AverageProfessorRating", n_boot=500, alpha=0.005)
            if np.isnan(lb):
                st.write("Insufficient data for bootstrap.")
            else:
                st.write(f"95% CI => [{lb:.3f}, {ub:.3f}], mean={mean_:.3f}")

    # ========================= Q4 ==========================
    elif question_choice == "Q4: Tag Analysis (Proportions) by Gender":
        st.header("Q4: Tag Analysis (Proportions) by Gender")
        st.markdown("""
        **Question 4** explores which tags differ by gender.  
        - Each tag is normalized by dividing by # of Ratings.  
        - We can do Mann-Whitney & KS for each selected tag, and optionally plot one.  
        """)

        all_tags = list(df_tags_10plus.columns[8:])
        chosen_tags = st.multiselect(
            "Select Tag(s) to Compare for M vs F",
            all_tags,
            default=["Pop quizzes!"]
        )

        if chosen_tags:
            results = []
            for tg in chosen_tags:
                arr_m = df_tags_male[tg].dropna()
                arr_f = df_tags_female[tg].dropna()

                mw_stat, mw_p = mannwhitneyu(arr_m, arr_f)
                mw_sig = "Significant" if mw_p < 0.005 else "Not Significant"

                ks_stat, ks_p = ks_2samp(arr_m, arr_f)
                ks_sig = "Significant" if ks_p < 0.005 else "Not Significant"

                results.append({
                    "Tag": tg,
                    "MW_stat": f"{mw_stat:.4f}",
                    "MW_p": f"{mw_p:.3g}",
                    "MW_Result": mw_sig,
                    "KS_stat": f"{ks_stat:.4f}",
                    "KS_p": f"{ks_p:.3g}",
                    "KS_Result": ks_sig
                })

            df_res = pd.DataFrame(results)
            st.write("### Test Results (Tag Analysis)")
            st.dataframe(df_res)

            # Distribution plot for one tag
            if len(chosen_tags) == 1:
                single_tag = chosen_tags[0]
            else:
                single_tag = st.selectbox("Choose ONE tag to visualize:", chosen_tags)

            distribution_plot_tag(df_tags_male, df_tags_female, single_tag)
        else:
            st.info("No tags selected. Please pick at least one tag above.")

    # ========================= Q5 ==========================
    elif question_choice == "Q5: Compare Average Difficulty by Gender":
        st.header("Q5: Compare Average Difficulty by Gender")
        st.markdown("""
        **Question 5** checks if M vs F differ in *Average Difficulty*.  
        - We do KS & MW tests, optionally filtering by Pepper or #Ratings≥19.  
        """)

        # Subset choice
        subset_choice = st.selectbox(
            "Subset for M vs F (Average Difficulty)",
            ["All (≥10 Ratings)", "Pepper=Yes", "Pepper=No", "≥19 Ratings"]
        )

        df_m = df_male.copy()
        df_f = df_female.copy()
        if subset_choice == "Pepper=Yes":
            df_m = df_m[df_m['Received a pepper'] == 1]
            df_f = df_f[df_f['Received a pepper'] == 1]
        elif subset_choice == "Pepper=No":
            df_m = df_m[df_m['Received a pepper'] == 0]
            df_f = df_f[df_f['Received a pepper'] == 0]
        elif subset_choice == "≥19 Ratings":
            df_m = df_m[df_m['NumberOfRatings'] >= 19]
            df_f = df_f[df_f['NumberOfRatings'] >= 19]

        st.subheader("Distribution Plot: Average Difficulty")
        distribution_plot(df_m, df_f, "Average Difficulty")

        st.subheader("KS & MW Tests")
        (ks_stat, ks_p, ks_str), (mw_stat, mw_p, mw_str) = ks_mw_test(df_m, df_f, "Average Difficulty", alpha=0.005)
        st.write(f"**KS**: stat={ks_stat:.4f}, p={ks_p:.3g} → {ks_str}")
        st.write(f"**MW**: stat={mw_stat:.4f}, p={mw_p:.3g} → {mw_str}")

    # ========================= Q6 ==========================
    else:
        st.header("Q6: Effect Size for Average Difficulty by Pepper Subgroups")
        st.markdown("""
        **Question 6** examines effect sizes for *Average Difficulty* specifically in 
        Pepper=Yes vs Pepper=No subgroups.  
        We compute Cohen's d for each subgroup, then optionally do a bootstrap CI.
        """)

        df_m_yes = df_male[df_male['Received a pepper'] == 1]
        df_f_yes = df_female[df_female['Received a pepper'] == 1]
        df_m_no  = df_male[df_male['Received a pepper'] == 0]
        df_f_no  = df_female[df_female['Received a pepper'] == 0]

        d_yes = cohen_d(df_m_yes, df_f_yes, "Average Difficulty")
        d_no  = cohen_d(df_m_no,  df_f_no,  "Average Difficulty")

        st.write("**Cohen's d for Average Difficulty**")
        st.table({
            "Pepper Subgroup": ["Yes", "No"],
            "Cohen's d": [f"{d_yes:.3f}", f"{d_no:.3f}"]
        })

        if st.button("Bootstrap Confidence Intervals"):
            st.subheader("Pepper=Yes")
            lb_y, ub_y, mean_y = bootstrap_cohen_d(df_m_yes, df_f_yes, "Average Difficulty", alpha=0.005)
            if np.isnan(lb_y):
                st.write("Insufficient data for Pepper=Yes.")
            else:
                st.write(f"95% CI => [{lb_y:.3f}, {ub_y:.3f}], mean={mean_y:.3f}")

            st.subheader("Pepper=No")
            lb_n, ub_n, mean_n = bootstrap_cohen_d(df_m_no, df_f_no, "Average Difficulty", alpha=0.005)
            if np.isnan(lb_n):
                st.write("Insufficient data for Pepper=No.")
            else:
                st.write(f"95% CI => [{lb_n:.3f}, {ub_n:.3f}], mean={mean_n:.3f}")

    st.write("---")
    st.info("You can switch questions from the sidebar and re-run as desired.")

# Standard script entry
if __name__ == "__main__":
    main()
