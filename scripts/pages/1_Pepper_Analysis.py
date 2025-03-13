import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# =============  CUSTOM UTILITY FUNCTIONS HERE =================
# - visualize_density_plot
# - perform_ks_mw_test
# - lavenes_test
# - effect_size
# ==================================================================

def visualize_density_plot(df1, df2, column, str1, str2, df3=None, str3=None, nbins=30):
    """Example version that uses st.pyplot instead of plt.show()."""
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df1[column], bins=nbins, kde=True, label=str1, stat='density')
    sns.histplot(df2[column], bins=nbins, kde=True, label=str2, stat='density')
    if df3 is not None:
        sns.histplot(df3[column], bins=nbins, kde=True, label=str3, stat='density')
        ax.set_title(f'{column}: {str1} vs {str2} vs {str3}')
    else:
        ax.set_title(f'{column}: {str1} vs {str2}')
    ax.legend()
    st.pyplot(fig)

def perform_ks_mw_test(df1, df2, column, label1, label2, alpha=0.005):
    """
    Performs KS and Mann-Whitney on column from df1 & df2, prints interpretation for alpha=0.005.
    """
    ks_stat, ks_p = stats.ks_2samp(df1[column], df2[column])
    u_stat, mw_p = stats.mannwhitneyu(df1[column], df2[column])

    st.markdown(f"**KS Test** for {label1} vs. {label2}, Column={column}")
    st.write(f"  KS Statistic: {ks_stat:.4f}, P-value: {ks_p:.3g}")
    if ks_p < alpha:
        st.write(f"  Distributions differ (p<0.005).")
    else:
        st.write(f"  No significant difference (p≥0.005).")

    st.markdown(f"**Mann-Whitney U Test** for {label1} vs. {label2}, Column={column}")
    st.write(f"  U Statistic: {u_stat:.4f}, P-value: {mw_p:.3g}")
    if mw_p < alpha:
        st.write(f"  Locations (medians) differ (p<0.005).")
    else:
        st.write(f"  No significant difference (p≥0.005).")

def lavenes_test(df1, df2, column, group1_name, group2_name, alpha=0.005):
    """
    Levene's test for difference in variance.
    """
    from scipy.stats import levene
    stat, p_value = levene(df1[column], df2[column])
    st.markdown(f"**Levene's Test** for {group1_name} vs. {group2_name}, Column={column}")
    st.write(f"  Statistic: {stat:.4f}, P-value: {p_value:.3g}")
    if p_value < alpha:
        st.write(f"  Variances differ (p<0.005).")
    else:
        st.write(f"  No significant difference in variance (p≥0.005).")

def effect_size(df1, df2, column):
    """
    Simple Cohen's d effect size for two groups (df1 & df2 on the same column).
    """
    mean_diff = df1[column].mean() - df2[column].mean()
    pooled_std = np.sqrt((df1[column].var() + df2[column].var())/2)
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0
    return cohen_d

# ============= THE STREAMLIT APP =================
def main():
    warnings.filterwarnings("ignore")
    st.title("Assessing Professor Effectiveness (APE) – Questions 1 to 6")

    #
    # LOAD & PREPROCESS
    #
    st.header("Data Loading & Basic Preprocessing")
    df_capstone = pd.read_csv('./../data/rmpCapstoneNum.csv', header=0)
    df_capstone.columns = [
        'AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper',
        'Proportion of students that said they would take the class again',
        'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale'
    ]

    # Let’s do the subset with ≥10 ratings:
    df_10plus = df_capstone[df_capstone['NumberOfRatings'] >= 10].copy()
    # Filter out rows that have both male & female = 1 or 0 (so exactly one gender)
    df_10plus = df_10plus[~(
        ((df_10plus['HighConfMale'] == 1) & (df_10plus['HighConfFemale'] == 1)) |
        ((df_10plus['HighConfMale'] == 0) & (df_10plus['HighConfFemale'] == 0))
    )].copy()

    df_10plus_male   = df_10plus[df_10plus['HighConfMale'] == 1]
    df_10plus_female = df_10plus[df_10plus['HighConfFemale'] == 1]

    st.write(f"**Overall shape (≥10 ratings, one-gender rows)**: {df_10plus.shape}")

    #
    # QUESTION 1
    #
    st.subheader("Q1: Is there a difference in the distribution / median of Average Ratings by Professor Gender?")
    st.markdown("""\
    **Hypothesis** (from the report):
    - Null: The distribution/location of M vs F average ratings is the same.
    - Alt: There is a difference in distribution/location.  

    In the report, we use **Kolmogorov-Smirnov (KS)** to check distribution difference and 
    **Mann-Whitney U (MWU)** to check median difference. We also consider 2 confounds:
    (1) *Pepper* (hotness, from Wallisch & Cachia, 2018)  
    (2) *Years of experience* (proxy: # of Ratings, from Centra & Gaubatz, 2000).
    """)

    # Basic test M vs F
    st.markdown("**1. Compare All M vs. F (≥10 ratings)**")
    visualize_density_plot(df_10plus_male, df_10plus_female,
                           'AverageProfessorRating', 'Male≥10', 'Female≥10')
    perform_ks_mw_test(df_10plus_male, df_10plus_female,
                       'AverageProfessorRating', 'Male≥10', 'Female≥10')

    st.markdown("""
    *Report Excerpt:*  
    "KS test returns a p-value of ~2.8e-3, MWU returns ~7.3e-4, both < α=0.005. We reject
    the null that M and F have the same distribution/median for average ratings." 
    """)

    # Next: pepper & # of ratings as potential confounds
    st.markdown("**2. Check Pepper as a Potential Confound**")
    pepper_yes = df_10plus[df_10plus['Received a pepper'] == 1]
    pepper_no  = df_10plus[df_10plus['Received a pepper'] == 0]

    st.markdown("_Compare distribution of AverageProfessorRating for Pepper=Yes vs. Pepper=No:_")
    visualize_density_plot(pepper_yes, pepper_no, 'AverageProfessorRating', 'Pepper=1', 'Pepper=0')
    perform_ks_mw_test(pepper_yes, pepper_no, 'AverageProfessorRating', 'Pepper=1', 'Pepper=0')

    st.markdown("""
    *Report Excerpt:*  
    "The difference in distributions for Pepper vs. Non-pepper is 'blaringly clear'. 
    KS p-value ~2.4e-322, MWU p=0.0 => Pepper significantly affects average ratings."
    """)

    st.markdown("**3. Check # of Ratings (Years of Experience) as a Potential Confound**")
    # Let's make 3 groups: 10-12, 13-18, and 19+
    df_10_12 = df_10plus[(df_10plus['NumberOfRatings'] >= 10) & (df_10plus['NumberOfRatings'] <= 12)]
    df_13_18 = df_10plus[(df_10plus['NumberOfRatings'] >= 13) & (df_10plus['NumberOfRatings'] <= 18)]
    df_19    = df_10plus[df_10plus['NumberOfRatings'] > 18]

    st.markdown("_Density Plot (10-12 vs. 13-18 vs. 19+)_")
    visualize_density_plot(df_10_12, df_13_18, 'AverageProfessorRating', '10-12', '13-18',
                           df3=df_19, str3='19+')

    st.markdown("""
    *Report Excerpt (Kruskal-Wallis & Pairwise Tests):*  
    "KW test p=3.2e-3 => at least one group differs. The 19+ group is different from the other two 
    while 10-12 vs 13-18 is not significantly different."
    """)

    # After adjusting for confounds, only "19+ NoPepper" was still significant M vs F
    st.markdown("""\
    After controlling for these confounds, we find that only in one subgroup 
    (19+ ratings, No Pepper) do M vs F remain significantly different. 
    """)

    # 
    # QUESTION 2
    #
    st.subheader("Q2: Is there a difference in the *variance* (spread) of Average Ratings by Gender?")
    st.markdown("""
    **Hypothesis**:  
    - Null: There's NO difference in the variance of M vs. F average ratings.  
    - Alt: M vs. F have different variances in average ratings.  

    We use **Levene's Test** to compare variance. The initial test (no confounds) gave p=0.0024, 
    indicating a difference in variance. But we next considered confounds (Number of Ratings, 
    Average Difficulty, Pepper) as each significantly influences the variance. 
    """)

    st.markdown("**Initial Levene's test for M vs. F**:")
    lavenes_test(df_10plus_male, df_10plus_female,
                 'AverageProfessorRating', "Male≥10", "Female≥10")

    st.markdown("""
    *Report Excerpt:*  
    "P-value=0.0024 < 0.005 => significant difference.  
     However, after controlling for confounds, only one subgroup 
     (Below median difficulty, below median # ratings, pepper=Yes) 
     had a significant difference in variance, with a very low power (0.043).  
     Conclusion: No broad difference in variance after controlling."  
    """)

    #
    # QUESTION 3
    #
    st.subheader("Q3: Confidence Intervals & Effect Sizes for M vs. F Differences in Average Ratings")
    st.markdown("""
    The project computed 95% confidence intervals for Cohen's d in various subgroups. 
    We highlight two main effect sizes:
    1. M vs F (≥10 ratings, no confound control) => 
       - Mean effect size ~0.086  
       - 95% CI: [0.043, 0.136]

    2. M vs F (19+ ratings, No Pepper) => 
       - Mean effect size ~0.27  
       - 95% CI: [0.13, 0.404]  

    The second group's effect is bigger but the power was only ~0.15 (quite low).
    """)

    # Example effect size
    es_10plus = effect_size(df_10plus_male, df_10plus_female, 'AverageProfessorRating')
    st.write(f"**Estimated Cohen's d (≥10 M vs. F)**: {es_10plus:.3f} (demo calculation)")

    st.markdown("""
    *Report Excerpt:*  
    "We see that after confound control, we get an effect size 
    of ~0.27 in the 19+ NoPepper group (95% CI ~ [0.13,0.40]), 
    but low power (0.15) implies caution in concluding a stable effect." 
    """)

    #
    # QUESTION 4
    #
    st.subheader("Q4: Which 'tags' differ by Gender?")
    st.markdown("""
    - Each 'tag' is how often a certain descriptor (e.g. *Tough grader*) was used. 
    - Normalized by dividing #tag for a professor by #ratings they received.
    - We do Mann-Whitney (MW) and KS for each tag M vs F. 
    - Among 10+ ratings, no pepper: 'Pop quizzes!' had p=0.0017 in MWU but not in KS, 
      so partially significant.  

    *Report Excerpt:*  
    "The only result that repeatedly appeared as statistically significant across 
    both main and confound-adjusted data was the 'Pop quizzes!' tag."  
    """)

    st.markdown("_(The actual code for tags is omitted here but you would do the same approach, merging tags DataFrame, computing p-values, etc.)_")

    #
    # QUESTION 5 & 6
    #
    st.subheader("Q5 & Q6: Does Average Difficulty differ by Gender?")
    st.markdown("""
    **Question**: Is there a difference in *Average Difficulty* (like Q1 but for difficulty) 
    between M vs. F?  
    **Findings**:
    1. The initial test with no confounds gave MW p=0.786, KS p=0.997 => *not significant*.
    2. We suspected 'Pepper' might confound difficulty. So we split by Pepper=Yes/No. 
       - Still no difference in M vs F within those subgroups.
    3. The effect size was extremely small (d ~ 0.05) with wide intervals. 
       The test had very low power (0.005–0.05).  

    **Conclusion**: No evidence of a consistent difference in Average Difficulty 
    between male vs. female professors.
    """)

    st.markdown("""
    *Report Excerpt:*  
    "We found no statistically significant difference in average difficulty 
    among men and women. Controlling for Pepper also yielded no difference. 
    The effect size was near zero, with extremely low power."  
    """)

    st.write("**Example short demonstration** (KS & MW for difficulty, no confounds):")
    male_diff = df_10plus_male['Average Difficulty']
    female_diff = df_10plus_female['Average Difficulty']
    ks_stat, ks_p = stats.ks_2samp(male_diff, female_diff)
    u_stat, mw_p = stats.mannwhitneyu(male_diff, female_diff)

    st.write(f"- KS p-value = {ks_p:.3g}")
    st.write(f"- MWU p-value = {mw_p:.3g}")
    if mw_p < 0.005:
        st.write("Significant difference. (But actual analysis found no difference, presumably large p).")
    else:
        st.write("No significant difference. (Matches the final conclusion: no difference in difficulty.)")


    st.markdown("### Final Remarks on Q1–Q6")
    st.markdown("""
    1. **Q1**: M vs F differ in average rating distributions, but after controlling 
       for pepper & #ratings, significance remains only in 19+ NoPepper. Low power though.  
    2. **Q2**: M vs F differ in variance initially, but not after confound controls.  
    3. **Q3**: Cohen's d effect sizes are generally small (~0.086) except 
       for 19+ NoPepper group (~0.27) with low power.  
    4. **Q4**: Tag usage. "Pop quizzes!" was the only repeated 'significant' difference.  
    5. & 6. **Avg Difficulty** does not differ by gender, even controlling for pepper.  
    """)

    st.info("For more details, see the reference PDF. Additional sections (Q7–Q11) can be handled similarly.")

if __name__ == "__main__":
    main()
