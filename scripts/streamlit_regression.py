import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_utils import PepperAnalysis, RegressionAnalysis

# Optional: set page config for better layout
st.set_page_config(
    page_title="Pepper Analysis and Regression Exploration",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(
    use_preloaded: bool, 
    preloaded_main_path: str = "./data/rmpCapstoneNum.csv", 
    preloaded_tags_path: str = "./data/rmpCapstoneTags.csv"
):
    """
    Load the dataset(s) based on the user's choice: preloaded or uploaded.
    Returns:
       df_capstone (pd.DataFrame): main numeric dataset
       tagsdf (pd.DataFrame): tags dataset
    """
    if use_preloaded:
        df_capstone = pd.read_csv(preloaded_main_path)
        tagsdf = pd.read_csv(preloaded_tags_path)
        st.success("Loaded preloaded CSV files.")
    else:
        uploaded_file_capstone = st.file_uploader(
            "Upload the main capstone CSV (e.g. rmpCapstoneNum.csv):", 
            type=["csv"]
        )
        uploaded_file_tagsdf = st.file_uploader(
            "Upload the tags CSV (e.g. rmpCapstoneTags.csv):", 
            type=["csv"]
        )
        if not uploaded_file_capstone or not uploaded_file_tagsdf:
            st.warning("Please upload both data files to proceed.")
            return None, None
        df_capstone = pd.read_csv(uploaded_file_capstone)
        tagsdf = pd.read_csv(uploaded_file_tagsdf)
        st.success("Successfully loaded your CSV files.")

    # Clean column names
    df_capstone.columns = [
        'AverageProfessorRating', 
        'Average Difficulty', 
        'NumberOfRatings', 
        'Received a pepper', 
        'Proportion of students that said they would take the class again', 
        'Number of ratings coming from online classes', 
        'HighConfMale', 
        'HighConfFemale'
    ]
    tagsdf.columns = list(range(20))
    return df_capstone, tagsdf


def show_data_previews(df_capstone: pd.DataFrame, tagsdf: pd.DataFrame):
    """
    Display head previews of capstone & tags dataframes.
    """
    st.markdown("### Preview of Capstone Data")
    st.dataframe(df_capstone.head(5))

    st.markdown("### Preview of Tags Data")
    st.dataframe(tagsdf.head(5))


def run_pepper_analysis(df_capstone: pd.DataFrame, tagsdf: pd.DataFrame):
    """
    Run Pepper Analysis steps:
      - Preprocess
      - Show correlation matrix & scatter plot
      - Single & multi-variable logistic regression
      - SVM training
    """
    st.markdown("## Pepper Analysis (Logistic & SVM)")

    if "pepper_analysis_clicked" not in st.session_state:
        st.session_state.pepper_analysis_clicked = False

    def click_button():
        st.session_state.pepper_analysis_clicked = True

    st.button("Run Pepper Analysis", on_click=click_button)

    if st.session_state.pepper_analysis_clicked:
        analysis = PepperAnalysis(df_capstone, tagsdf, seed=42)
        analysis.preprocess_data()
        st.success("Data Preprocessed (inner join, dropna, proportions, M/F filters).")

        # Correlation & Scatter Plots in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Correlation Matrix")
            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
            corr_matrix = analysis.df.corr()
            sns.heatmap(corr_matrix, cmap="RdBu_r", annot=True, ax=ax_corr)
            st.pyplot(fig_corr)

        with col2:
            st.markdown("#### Scatter Plot (AverageProfessorRating vs. Received a pepper)")
            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            ax_scatter.scatter(
                analysis.df['AverageProfessorRating'], 
                analysis.df['Received a pepper']
            )
            ax_scatter.set_xlabel("AverageProfessorRating")
            ax_scatter.set_ylabel("Received a pepper")
            ax_scatter.set_title("Scatter Plot")
            st.pyplot(fig_scatter)

        # Single Variable Logistic Regression
        st.markdown("### Single-variable Logistic Regression")
        target_col = 'Received a pepper'
        all_features = [
            col for col in analysis.df.columns 
            if col != target_col and not col.isdigit()
        ]
        selected_feature = st.selectbox(
            "Select a single feature:", 
            options=all_features
        )

        if st.button("Run Single-Feature Logistic Regression"):
            with st.spinner("Running single-var logistic regression..."):
                fig1, fig2, fig3 = analysis.logistic_regression_single_var(
                    x_col=selected_feature,
                    y_col='Received a pepper',
                    threshold=0.607  # example threshold
                )
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)

        # Multi Variable Logistic Regression
        st.markdown("### Multi-variable Logistic Regression")
        selected_features_multi = st.multiselect(
            "Select multiple features:", 
            options=all_features, 
            default=all_features, 
            key="multi_logistic"
        )
        if st.button("Run Multi-Feature Logistic Regression"):
            drop_cols = set(analysis.df.columns) - set(selected_features_multi)
            with st.spinner("Running multi-var logistic regression..."):
                fig1, fig2 = analysis.logistic_regression_multi_var(
                    threshold=0.465, 
                    drop_cols=list(drop_cols)
                )
            st.pyplot(fig1)
            st.pyplot(fig2)

        # Train a linear SVM
        st.markdown("### Train a Linear SVM")
        selected_features_svm = st.multiselect(
            "Select features for the SVM:", 
            options=all_features, 
            default=all_features, 
            key="multi_svm"
        )
        if st.button("Run SVM"):
            drop_cols_svm = set(analysis.df.columns) - set(selected_features_svm)
            with st.spinner("Training SVM..."):
                fig_svm = analysis.train_svm(drop_cols=list(drop_cols_svm))
            st.pyplot(fig_svm)


def run_regression_analysis(df_capstone: pd.DataFrame, tagsdf: pd.DataFrame):
    """
    Run the RegressionAnalysis steps on subsets of features 
    (by default just once, with user-selected features).
    """
    st.markdown("## Regression Analysis on Subsets")

    # Prepare data
    analysis = PepperAnalysis(df_capstone, tagsdf, seed=42)
    analysis.preprocess_data()

    target_col = 'AverageProfessorRating'
    all_features = [
        col for col in analysis.df.columns 
        if col != target_col and not col.isdigit() and col != "HighConfMale"
    ]

    selected_features = st.multiselect(
        "Select features for the regression model:", 
        options=all_features, 
        default=all_features.drop,
        help="These features will be used to predict AverageProfessorRating."
    )

    if st.button("Run Subset RegressionAnalysis"):
        if not selected_features:
            st.warning("Please select at least one feature to proceed!")
            return

        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(analysis.df, test_size=0.2, random_state=42)

        # We only do one subset here, but you could loop multiple if desired.
        st.write(f"**Selected Features:** {selected_features}")

        # Prepare X/y
        X_train = df_train[selected_features].to_numpy()
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[selected_features].to_numpy()
        y_test = df_test[target_col].to_numpy()

        alphas = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 5, 10, 20, 100, 1000])

        reg_analysis = RegressionAnalysis(
            X_train, y_train,
            alphas=alphas,
            seed=42,
            n_splits=5
        )

        # Cross-validate
        reg_analysis.cross_validate()
        cv_results_df = reg_analysis.get_cv_results_df()

        st.markdown("### Cross-Validation Results")
        st.dataframe(cv_results_df.head(10))

        st.markdown("### Cross-Validation Performance (Plots)")
        fig_rmse, fig_r2 = reg_analysis.plot_cv_rmse_r2()
        st.pyplot(fig_rmse)
        st.pyplot(fig_r2)

        # Pick best alpha
        best_ridge_alpha, best_ridge_rmse = reg_analysis.pick_best_alpha("Ridge", metric="RMSE")
        best_lasso_alpha, best_lasso_rmse = reg_analysis.pick_best_alpha("Lasso", metric="RMSE")
        st.write(f"**Best Ridge alpha**: {best_ridge_alpha:.5f} | RMSE = {best_ridge_rmse:.3f}")
        st.write(f"**Best Lasso alpha**: {best_lasso_alpha:.5f} | RMSE = {best_lasso_rmse:.3f}")

        # Finalize & evaluate on the test set
        residualplot = reg_analysis.finalize_and_evaluate(
            X_train, y_train,
            X_test, y_test,
            best_ridge_alpha,
            best_lasso_alpha,
            make_residual_plots=True
        )
        st.pyplot(residualplot)

        final_test_df = reg_analysis.get_test_results_df()
        st.markdown("### Final Test Results")
        st.dataframe(final_test_df)

        st.markdown("### Model Coefficients")
        normal_row = final_test_df[final_test_df['Model'] == 'Normal'].iloc[0]
        betas_normal = normal_row['Betas']

        st.write("**Normal Coefficients**")
        coefplot = reg_analysis.plot_coefs(
            betas_normal[1:],  # skip intercept
            feature_names=selected_features,
            model_name='Normal'
        )
        st.pyplot(coefplot)

        st.success("Regression Analysis complete!")


def main():
    """Main function to run the Streamlit app."""
    st.title("Pepper Analysis and Regression Exploration")

    with st.expander("1) Data Upload / Selection", expanded=True):
        st.write("Choose whether to use the preloaded datasets or upload your own.")
        option = st.radio(
            "Choose an option:", 
            ("Use Preloaded Datasets", "Upload Your Own CSV")
        )
        use_preloaded = (option == "Use Preloaded Datasets")
        df_capstone, tagsdf = load_data(use_preloaded=use_preloaded)
        
        if df_capstone is not None and tagsdf is not None:
            show_data_previews(df_capstone, tagsdf)
        else:
            st.stop()  # End execution if data not available

    with st.expander("2) Pepper Analysis (Logistic & SVM)", expanded=False):
        run_pepper_analysis(df_capstone, tagsdf)

    with st.expander("3) Regression Analysis on Subsets", expanded=False):
        run_regression_analysis(df_capstone, tagsdf)


if __name__ == "__main__":
    main()
