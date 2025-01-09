
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page
st.set_page_config(page_title="Density Plot Visualizer", layout="wide")

# Title and description
st.title("Density Plot Visualizer")
st.write("Upload your datasets to compare density plots for specific columns.")

# File upload section
st.sidebar.header("Upload Data")
uploaded_file1 = st.sidebar.file_uploader("Upload first dataset (CSV)", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload second dataset (CSV)", type=["csv"])

# Helper function to visualize density plot
def visualize_density_plot(df1, df2, column, str1, str2, nbins=30):
    plt.figure(figsize=(10, 6))
    sns.histplot(df1[column], kde=True, bins=nbins, label=str1, color='blue', alpha=0.5)
    sns.histplot(df2[column], kde=True, bins=nbins, label=str2, color='red', alpha=0.5)
    plt.legend(title="Datasets")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Density Plot for {column}")
    st.pyplot(plt)

# Data loading and processing
if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)
    
    st.sidebar.header("Plot Configuration")
    column_options = list(set(df1.columns) & set(df2.columns))  # Common columns for comparison
    
    if column_options:
        selected_column = st.sidebar.selectbox("Select column for density plot", column_options)
        label1 = st.sidebar.text_input("Label for first dataset", "Dataset 1")
        label2 = st.sidebar.text_input("Label for second dataset", "Dataset 2")
        bins = st.sidebar.slider("Number of bins", 10, 100, 30)
        
        # Visualize density plot
        st.subheader("Density Plot")
        visualize_density_plot(df1, df2, selected_column, label1, label2, nbins=bins)
    else:
        st.write("No common columns found between the two datasets.")
else:
    st.write("Please upload two CSV files to begin.")
