
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate effect size
def calculate_effect_size(data1, data2):
    mean_diff = np.mean(data1) - np.mean(data2)
    pooled_std = np.sqrt((np.std(data1, ddof=1) ** 2 + np.std(data2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std

# Bootstrap method to calculate confidence intervals
def bootstrap_effect_size(data1, data2, num_bootstrap=1000, ci=95):
    bootstrapped_effect_sizes = []
    for _ in range(num_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrapped_effect_sizes.append(calculate_effect_size(sample1, sample2))
    
    lower_bound = np.percentile(bootstrapped_effect_sizes, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_effect_sizes, 100 - (100 - ci) / 2)
    
    return lower_bound, upper_bound, bootstrapped_effect_sizes

# Streamlit app starts here
st.title("Statistical Analysis Web App")
st.markdown("Upload a dataset to calculate effect sizes and bootstrap confidence intervals.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.write(df.head())

    # Select columns for analysis
    column_options = df.columns.tolist()
    data1_col = st.selectbox("Select first group column", column_options)
    data2_col = st.selectbox("Select second group column", column_options)

    if data1_col and data2_col:
        data1 = df[data1_col].dropna()
        data2 = df[data2_col].dropna()

        st.write(f"Selected column for Group 1: {data1_col} ({len(data1)} values)")
        st.write(f"Selected column for Group 2: {data2_col} ({len(data2)} values)")

        # Calculate effect size and bootstrap confidence intervals
        lower_bound, upper_bound, bootstrapped_effect_sizes = bootstrap_effect_size(data1, data2)

        st.write(f"Effect Size: {calculate_effect_size(data1, data2):.3f}")
        st.write(f"95% Confidence Interval for Effect Size: [{lower_bound:.3f}, {upper_bound:.3f}]")

        # Plot the bootstrap distribution
        st.write("Bootstrap Distribution of Effect Sizes:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(bootstrapped_effect_sizes, bins=30, kde=True, ax=ax)
        ax.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound: {lower_bound:.3f}')
        ax.axvline(upper_bound, color='green', linestyle='--', label=f'Upper Bound: {upper_bound:.3f}')
        ax.set_title('Bootstrap Distribution of Effect Sizes')
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
