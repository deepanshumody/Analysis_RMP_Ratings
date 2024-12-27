import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Seed value for reproducibility
RANDOM_SEED = 10676128
np.random.seed(RANDOM_SEED)

# Streamlit app setup
st.title("Professor Ratings Analysis App")
st.write("""
This app allows you to visualize and analyze professor ratings data.
Upload your CSV file, visualize the density plots, and perform statistical tests.
""")

# Function to visualize density plots
def visualize_density_plot(df1, df2, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df1[column], bins=30, kde=True, color='blue', label='Men', stat='density', ax=ax)
    sns.histplot(df2[column], bins=30, kde=True, color='red', label='Women', stat='density', ax=ax)
    ax.set_title('Density Plot of Average Professor Ratings')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)

# Function to perform KS and Mann-Whitney U tests
def perform_ks_mw_test(df1, df2, column):
    ks_stat, p_val = stats.ks_2samp(df1[column], df2[column])
    mannwhitney_stat, mannwhitney_p_val = stats.mannwhitneyu(df1[column], df2[column])

    st.subheader(f"Statistical Test Results for '{column}'")
    st.write(f"**Kolmogorov-Smirnov Test**")
    st.write(f"- Statistic: {ks_stat}")
    st.write(f"- P-value: {p_val}")
    st.write(f"**Mann-Whitney U Test**")
    st.write(f"- Statistic: {mannwhitney_stat}")
    st.write(f"- P-value: {mannwhitney_p_val}")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load the dataset
    df_capstone = pd.read_csv(uploaded_file)
    df_capstone.columns = [
        'AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 
        'Received a pepper', 'Proportion of students that said they would take the class again', 
        'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale'
    ]

    # Data preprocessing
    df_filtered = df_capstone[
        (df_capstone['NumberOfRatings'] >= 10) &
        ~((df_capstone['HighConfMale'] == 1) & (df_capstone['HighConfFemale'] == 1)) &
        ~((df_capstone['HighConfMale'] == 0) & (df_capstone['HighConfFemale'] == 0))
    ]

    df_men = df_filtered[df_filtered['HighConfMale'] == 1]
    df_women = df_filtered[df_filtered['HighConfFemale'] == 1]

    # Show a preview of the data
    st.subheader("Filtered Dataset")
    st.write(df_filtered.head())

    # Dropdown to select column for analysis
    column = st.selectbox(
        "Select column for analysis",
        df_filtered.columns,
        index=list(df_filtered.columns).index('AverageProfessorRating')
    )

    # Visualize the density plot
    visualize_density_plot(df_men, df_women, column)

    # Perform statistical tests
    perform_ks_mw_test(df_men, df_women, column)
else:
    st.info("Awaiting CSV file to be uploaded.")
