import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import *only* the classes and functions you need from another file.
# For example, if you have a file called "analysis_utils.py" that defines:
#   - PepperAnalysis
#   - RegressionAnalysis
#   - plot_feature_vs_dependent
# ...then import them here:
from analysis_utils import PepperAnalysis, RegressionAnalysis, plot_feature_vs_dependent


def main():
    st.title("Pepper Analysis and Regression Exploration")

    # --------------------------------------------------------------
    # 1) Data Upload / Selection Section
    # --------------------------------------------------------------
    st.subheader("Upload Your Datasets")

    uploaded_file_capstone = st.file_uploader(
        "Upload the main capstone CSV (e.g. df_capstone) here", 
        type=["csv"]
    )
    uploaded_file_tagsdf = st.file_uploader(
        "Upload the tags CSV (e.g. tagsdf) here", 
        type=["csv"]
    )

    if uploaded_file_capstone is not None and uploaded_file_tagsdf is not None:
        # Read data into DataFrames
        df_capstone = pd.read_csv(uploaded_file_capstone)
        tagsdf = pd.read_csv(uploaded_file_tagsdf)
        df_capstone.columns = ['AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper', 
                        'Proportion of students that said they would take the class again', 
                        'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale']
        tagsdf.columns = list(range(20))

        st.write("**Preview of df_capstone:**")
        st.dataframe(df_capstone.head(5))

        st.write("**Preview of tagsdf:**")
        st.dataframe(tagsdf.head(5))

        # --------------------------------------------------------------
        # 2) Run Pepper Analysis
        # --------------------------------------------------------------
        st.subheader("Pepper Analysis (Logistic & SVM)")
        if st.button("Run Pepper Analysis"):
            # 2.1 Create the PepperAnalysis instance
            analysis = PepperAnalysis(df_capstone, tagsdf, seed=42)

            # 2.2 Preprocess
            analysis.preprocess_data()
            st.success("Data Preprocessed (inner join, dropna, proportions, male/female filter).")

            # 2.3 Optional: Correlation Matrix
            st.write("**Correlation Matrix**")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            corr_matrix = analysis.df.corr()
            sns.heatmap(corr_matrix, cmap="RdBu_r", annot=True, ax=ax_corr)
            st.pyplot(fig_corr)

            # 2.4 Optional: Scatter Plot
            st.write("**Scatter Plot (AverageProfessorRating vs. Received a pepper)**")
            fig_scatter, ax_scatter = plt.subplots(figsize=(6,4))
            ax_scatter.scatter(
                analysis.df['AverageProfessorRating'], 
                analysis.df['Received a pepper'], 
                color='purple'
            )
            ax_scatter.set_xlabel("AverageProfessorRating")
            ax_scatter.set_ylabel("Received a pepper")
            ax_scatter.set_title("Scatter: AverageProfessorRating vs. Received a pepper")
            st.pyplot(fig_scatter)

            # Alternatively, you could call analysis.plot_scatter_single(...) 
            # But that function prints/plots directly.
            # For example:
            # analysis.plot_scatter_single()

            # 2.5 Single-variable Logistic Regression
            st.write("**Single-variable Logistic Regression**")
            with st.spinner("Running single-var logistic regression..."):
                # This method may print results to stdout, which Streamlit won't capture automatically.
                # You can either let it print to your console,
                # or refactor the method to return values that you display with st.write().
                analysis.logistic_regression_single_var(
                    x_col='AverageProfessorRating',
                    y_col='Received a pepper',
                    threshold=0.607
                )
            st.success("Single-variable logistic regression complete!")

            # 2.6 Multi-variable logistic regression
            st.write("**Multi-variable Logistic Regression**")
            with st.spinner("Running multi-var logistic regression..."):
                analysis.logistic_regression_multi_var(threshold=0.465)
            st.success("Multi-variable logistic regression complete!")

            # 2.7 Train a linear SVM
            st.write("**Train a linear SVM**")
            with st.spinner("Training SVM..."):
                analysis.train_svm()
            st.success("SVM training complete!")

        # --------------------------------------------------------------
        # 3) Run RegressionAnalysis on Different Feature Subsets
        # --------------------------------------------------------------
        st.subheader("RegressionAnalysis on Subsets")
        # Display checklist for feature selection
        analysis = PepperAnalysis(df_capstone, tagsdf, seed=42)
        analysis.preprocess_data()

        target_col = 'AverageProfessorRating'
        all_features = [col for col in analysis.df.columns if col != target_col and not col.isdigit()]
         # Streamlit interface
        st.title("Feature Selection for Linear Regression")
        selected_features = st.multiselect(
            "Select features for the regression model:", 
            options=all_features, 
            default=all_features  # Preselect all by default
        )
            
        if st.button("Run Subset RegressionAnalysis"):

            if not selected_features:
                st.warning("Please select at least one feature to proceed!")
            else:
                st.write(f"Selected Features: {selected_features}")
                # Train-test split
                from sklearn.model_selection import train_test_split
                df_train, df_test = train_test_split(analysis.df, test_size=0.2, random_state=42)

                all_subsets_results = []
                feature_subsets=[selected_features]
                for idx, subset_cols in enumerate(feature_subsets, start=1):
                    st.write(f"**Subset #{idx}:** {subset_cols}")

                    # 3.1 Prepare X_train, y_train, X_test, y_test
                    X_train = df_train[subset_cols].to_numpy()
                    y_train = df_train[target_col].to_numpy()
                    X_test = df_test[subset_cols].to_numpy()
                    y_test = df_test[target_col].to_numpy()

                    # 3.2 Instantiate RegressionAnalysis
                    alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 
                                    0.1, 1, 2, 5, 10, 20, 100, 1000])
                    
                    reg_analysis = RegressionAnalysis(
                        X_train, y_train,
                        alphas=alphas,
                        seed=42,
                        n_splits=5
                    )

                    # 3.3 Cross-validate
                    reg_analysis.cross_validate()
                    mydf=reg_analysis.get_cv_results_df()
                    st.dataframe(mydf.head(5))

                    # 3.4 Plot CV results
                    st.write("**Cross-Validation Performance**")
                    #fig_cv = plt.figure(figsize=(12, 5))
                    fig_cv,fig_cv2=reg_analysis.plot_cv_rmse_r2()  # This method likely shows plots directly
                    st.pyplot(fig_cv)
                    st.pyplot(fig_cv2)

                    # 3.5 Pick best alphas
                    best_ridge_alpha, best_ridge_rmse = reg_analysis.pick_best_alpha('Ridge', metric='RMSE')
                    best_lasso_alpha, best_lasso_rmse = reg_analysis.pick_best_alpha('Lasso', metric='RMSE')
                    st.write(f"Best Ridge alpha: {best_ridge_alpha} (CV RMSE={best_ridge_rmse:.3f})")
                    st.write(f"Best Lasso alpha: {best_lasso_alpha} (CV RMSE={best_lasso_rmse:.3f})")

                    # 3.6 Finalize & Evaluate
                    residualplot=reg_analysis.finalize_and_evaluate(
                        X_train, y_train,
                        X_test, y_test,
                        best_ridge_alpha,
                        best_lasso_alpha,
                        make_residual_plots=True
                    )
                    st.pyplot(residualplot)

                    final_test_df = reg_analysis.get_test_results_df()
                    st.write("Final Test Results:")
                    st.dataframe(final_test_df)

                    # Optionally plot coefficients
                    # Example: Normal
                    normal_row = final_test_df[final_test_df['Model'] == 'Normal'].iloc[0]
                    betas_normal = normal_row['Betas']
                    st.write("**Normal Coefficients**")
                    coefplot=reg_analysis.plot_coefs(
                        betas_normal[1:], 
                        feature_names=subset_cols,
                        model_name=f'Normal (Subset #{idx})'
                    )
                    st.pyplot(coefplot)

                    # ... similarly for best Ridge/Lasso

                    all_subsets_results.append({
                        'subset_index': idx,
                        'subset_columns': subset_cols,
                        'test_results': final_test_df
                    })
                
                st.success("Subset processed!")

    else:
        st.warning("Please upload both data files to proceed.")


if __name__ == "__main__":
    main()
