##############################################
# streamlit_app_classification.py
# A Streamlit app for classification (Logistic & SVM)
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

class ClassificationAnalysis:
    """
    A class to handle classification tasks (Logistic Regression, SVM)
    for the RateMyProfessor Pepper data.
    """
    def __init__(self, df_capstone, tagsdf, seed=10676128):
        self.seed = seed
        self.df_capstone = df_capstone
        self.tagsdf = tagsdf
        self.df = None
        self.log_reg_single = None
        self.log_reg_multi = None
        self.svm_model = None

    def preprocess_data(self):
        # 1) Concatenate side by side
        Q10df = pd.concat([self.df_capstone, self.tagsdf], axis=1)

        # 2) Drop missing
        Q10df.dropna(inplace=True)

        # 3) Convert columns (index >=8) into proportion by dividing by #Ratings
        for col in Q10df.columns[8:]:
            Q10df[col] = Q10df[col] / Q10df['NumberOfRatings']

        # 4) Filter to exactly one gender
        #    (HighConfMale=1,HighConfFemale=0) or (HighConfMale=0,HighConfFemale=1)
        Q10df = Q10df[~(
            ((Q10df['HighConfMale'] == 1) & (Q10df['HighConfFemale'] == 1)) |
            ((Q10df['HighConfMale'] == 0) & (Q10df['HighConfFemale'] == 0))
        )].copy()

        self.df = Q10df.reset_index(drop=True)

    def plot_correlation_matrix(self):
        if self.df is None:
            st.warning("Data not preprocessed yet. Please run preprocess_data() first.")
            return
        corr_mat = self.df.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr_mat, cmap="RdBu_r", annot=False)
        plt.title("Correlation Matrix")
        st.pyplot(plt.gcf())

    def plot_scatter_single(self, x_col='AverageProfessorRating', y_col='Received a pepper'):
        if self.df is None:
            st.warning("Data not preprocessed yet.")
            return
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(self.df[x_col], self.df[y_col], c='purple', alpha=0.6)
        ax.set_title(f"Scatterplot: {x_col} vs. {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    def logistic_regression_single_var(self, x_col='AverageProfessorRating', y_col='Received a pepper', threshold=0.5):
        if self.df is None:
            st.error("Data not preprocessed. Please run preprocess_data() first.")
            return
        X = self.df[[x_col]].values
        y = self.df[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        self.log_reg_single = log_reg

        y_prob = log_reg.predict_proba(X_test)[:,1]
        y_pred = (y_prob > threshold).astype(int)

        st.write("### Single-Variable Logistic Regression Results")
        st.write(f"Using X={x_col}, threshold={threshold}")
        st.write("**Classification Report**:")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix**:")
        cmat = confusion_matrix(y_test, y_pred)
        st.write(cmat)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC: Single Var Logistic")
        ax.legend()
        st.pyplot(fig)

    def logistic_regression_multi_var(self, threshold=0.5, drop_cols=None):
        if self.df is None:
            st.error("Data not preprocessed. Please run preprocess_data() first.")
            return
        if drop_cols is None:
            drop_cols = [
                'Received a pepper', 'NumberOfRatings',
                'Number of ratings coming from online classes',
                'HighConfMale', 'HighConfFemale'
            ]
        y = self.df['Received a pepper']
        df_clean = self.df.drop(columns=drop_cols, errors='ignore')
        X = df_clean.select_dtypes(include=[np.number]).values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=self.seed)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        self.log_reg_multi = log_reg

        y_prob = log_reg.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)

        st.write("### Multi-Variable Logistic Regression")
        st.write(f"Threshold={threshold}")
        st.write("**Classification Report**:")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix**:")
        cmat = confusion_matrix(y_test, y_pred)
        st.write(cmat)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC: Multi-Var Logistic")
        ax.legend()
        st.pyplot(fig)

    def train_svm(self, drop_cols=None):
        if self.df is None:
            st.error("Data not preprocessed. Please run preprocess_data() first.")
            return
        if drop_cols is None:
            drop_cols = [
                'Received a pepper', 'NumberOfRatings',
                'Number of ratings coming from online classes',
                'HighConfMale', 'HighConfFemale'
            ]
        y = self.df['Received a pepper']
        df_clean = self.df.drop(columns=drop_cols, errors='ignore')
        X = df_clean.select_dtypes(include=[np.number]).values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=self.seed)
        svm_model = SVC(kernel='linear', probability=True, random_state=self.seed)
        svm_model.fit(X_train, y_train)
        self.svm_model = svm_model

        st.write("### SVM (Linear Kernel)")
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:,1]

        st.write("**Classification Report**:")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix**:")
        cmat = confusion_matrix(y_test, y_pred)
        st.write(cmat)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1], '--')
        ax.set_title("ROC Curve (SVM)")
        ax.legend()
        st.pyplot(fig)


# --------------------------------------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------------------------------------
def main():
    st.title("Classification App (Logistic & SVM)")

    # Initialize session_state entry for PepperAnalysis if not present
    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    st.markdown("""
    This demo shows a classification approach for the Pepper analysis, 
    focusing on logistic regression (single & multi) and SVM.
    """)

    st.header("1) Load Data & Instantiate PepperAnalysis")

    # Step 1: Provide data previews
    with st.expander("Preview the CSV Data"):
        df_capstone = pd.read_csv("./data/rmpCapstoneNum.csv", header=0)
        tagsdf = pd.read_csv("./data/rmpCapstoneTags.csv", header=0)
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
        st.subheader("First 5 rows of df_capstone:")
        st.dataframe(df_capstone.head())
        st.subheader("First 5 rows of tagsdf:")
        st.dataframe(tagsdf.head())

    seed_val = 10676128

    if st.button("Instantiate & Preprocess PepperAnalysis"):
        # Actually create PepperAnalysis, store in session_state
        analysis_obj = ClassificationAnalysis(df_capstone, tagsdf, seed=seed_val)
        analysis_obj.preprocess_data()
        st.session_state.analysis = analysis_obj
        st.success("PepperAnalysis created & data preprocessed!")
    else:
        st.info("Click above to create the PepperAnalysis object and run preprocessing.")

    st.header("2) Explore Data (Optional)")
    if st.session_state.analysis is not None and st.session_state.analysis.df is not None:
        # Show shape
        st.write(f"**Preprocessed Data Shape**: {st.session_state.analysis.df.shape}")
        if st.checkbox("Show a sample of the preprocessed data"):
            st.dataframe(st.session_state.analysis.df.head(10))
        if st.button("Correlation Matrix"):
            st.session_state.analysis.plot_correlation_matrix()
        if st.button("Scatter: AverageProfessorRating vs. Received a pepper"):
            st.session_state.analysis.plot_scatter_single()
    else:
        st.warning("PepperAnalysis not ready. Please instantiate & preprocess above.")

    st.header("3) Single-Variable Logistic Regression")
    if st.session_state.analysis is not None and st.session_state.analysis.df is not None:
        x_col = st.selectbox("Select Single X Column", 
                             ["AverageProfessorRating", "Average Difficulty", "NumberOfRatings"],
                             index=0)
        thr_sing = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
        if st.button("Run Single-Var Logistic"):
            st.session_state.analysis.logistic_regression_single_var(
                x_col=x_col,
                y_col='Received a pepper',
                threshold=thr_sing
            )
    else:
        st.warning("PepperAnalysis not ready. Please instantiate & preprocess above.")

    st.header("4) Multi-Variable Logistic Regression")
    if st.session_state.analysis is not None and st.session_state.analysis.df is not None:
        thr_multi = st.slider("Multi-Var Logistic Threshold", 0.0, 1.0, 0.5, 0.01)
        if st.button("Run Multi-Var Logistic"):
            st.session_state.analysis.logistic_regression_multi_var(threshold=thr_multi)
    else:
        st.warning("PepperAnalysis not ready. Please instantiate & preprocess above.")

    st.header("5) Train SVM (Linear)")
    if st.session_state.analysis is not None and st.session_state.analysis.df is not None:
        if st.button("Train SVM"):
            st.session_state.analysis.train_svm()
    else:
        st.warning("PepperAnalysis not ready. Please instantiate & preprocess above.")

    st.write("---")
    st.success("End of classification demonstration.")

if __name__ == "__main__":
    main()
