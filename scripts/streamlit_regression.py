import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import PepperAnalysis, RegressionAnalysis, plot_feature_vs_dependent


def main():
    st.title("Pepper Analysis and Regression Exploration")

    # --------------------------------------------------------------
    # 1) Data Upload / Selection Section
    # --------------------------------------------------------------
    st.subheader("Upload Your Datasets")

    uploaded_file_capstone = st.file_uploader(
        "Upload the main capstone CSV (e.g. rmpCapstoneNum) here", 
        type=["csv"]
    )
    uploaded_file_tagsdf = st.file_uploader(
        "Upload the tags CSV (e.g. rmpCapstoneTags) here", 
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

        st.write("**Preview of capstone:**")
        st.dataframe(df_capstone.head(5))

        st.write("**Preview of tags:**")
        st.dataframe(tagsdf.head(5))

        # --------------------------------------------------------------
        # 2) Run Pepper Analysis
        # --------------------------------------------------------------
        st.subheader("Pepper Analysis (Logistic & SVM)")
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        def click_button():
            st.session_state.clicked = True

        st.button("Run Pepper Analysis",on_click=click_button)
        if st.session_state.clicked:

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
            target_col = 'Received a pepper'
            all_features = [col for col in analysis.df.columns if col != target_col and not col.isdigit()]
            selected_feature = st.selectbox(
                "Select features for the regression model:", 
                options=all_features)
            if st.button("Run Single Feature Logistic Regression"):
                
                with st.spinner("Running single-var logistic regression..."):
                    fig1,fig2,fig3=analysis.logistic_regression_single_var(
                        x_col=selected_feature,
                        y_col='Received a pepper',
                        threshold=0.607
                    )
                st.pyplot(fig1)
                st.pyplot(fig2)
                st.pyplot(fig3)

            # 2.6 Multi-variable logistic regression
            selected_features = st.multiselect(
                "Select features for the regression model:", 
                options=all_features, 
                default=all_features,key='2'  # Preselect all by default
            )
            if st.button("Run Multi Feature Logistic Regression"):
                
                drop_cols=set(analysis.df.columns) - set(selected_features)
                with st.spinner("Running multi-var logistic regression..."):
                    fig1,fig2=analysis.logistic_regression_multi_var(threshold=0.465,drop_cols=list(drop_cols))
                st.pyplot(fig1)
                st.pyplot(fig2)

            # 2.7 Train a linear SVM
            selected_features_SVM = st.multiselect(
                "Select features for the regression model:", 
                options=all_features, 
                default=all_features, key='1'  # Preselect all by default
            )
            if st.button("Run Multi Feature SVM"):
                
                drop_cols=set(analysis.df.columns) - set(selected_features_SVM)
                with st.spinner("Training SVM..."):
                    fig1=analysis.train_svm(drop_cols=list(drop_cols))
                st.pyplot(fig1)

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
