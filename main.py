# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.model_selection import KFold  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Seed value for random number generators to obtain reproducible results
RANDOM_SEED = 10676128

# Apply the random seed to numpy.
np.random.seed(RANDOM_SEED)

def visualize_density_plot(df1, df2, column, str1, str2, df3 = None, str3 = None, nbins = 30):
    # Plot the histogram of the AverageProfessorRating for professors with more than 10 ratings for women and men separately
    plt.figure(figsize=(10, 6))

    sns.histplot(df1[column], bins=nbins, kde=True, color='blue', label=f'{column} for {str1}', stat='density')
    sns.histplot(df2[column], bins=nbins, kde=True, color='red', label=f'{column} for {str2}', stat='density')
    if(df3 is not None):
        sns.histplot(df3[column], bins=nbins, kde=True, color='green', label=f'{column} for {str3}', stat='density')
    if(df3 is not None):
        plt.title(f'Normalized Histogram of {str1}, {str2} and {str3} for {column}', fontsize=10)
    else:
        plt.title(f'Normalized Histogram of {str1} and {str2} for {column}', fontsize=10)
    plt.xlabel(f'{column}') 
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def perform_ks_mw_test(df1, df2, column, str1, str2):
    # Perform a KS test to check if the distributions of AverageProfessorRating for Male and Felame professors with more than 10 ratings are the same
    ks_stat, p_val = stats.ks_2samp(df1[column], df2[column])
    mannwhitney_stat, mannwhitney_p_val = stats.mannwhitneyu(df1[column], df2[column])
    print(f'KS Test of {column} for the two groups: {str1} and {str2}')
    print('KS Statistic: ', ks_stat)
    print('P-value: ', p_val)
    print('Mann Whitney U Statistic: ', mannwhitney_stat)
    print('Mann Whitney U P-value: ', mannwhitney_p_val)
    if(p_val < 0.005):
        print(f'We drop the null hypothesis and adopt: The distributions of {column} for {str1} and {str2} are different')
    else:
        print(f'We dont drop the null hypothesis and therefore retain that: The distributions of {column} for {str1} and {str2} are the same')
    if(mannwhitney_p_val < 0.005):
        print(f'We drop the null hypothesis and adopt: The median/location of {column} for {str1} and {str2} are different')
    else:
        if(p_val < 0.005):
            print(f'Since the distributions are different, despite a non-significant result from the Mann Whitney U test, we can say that the median/location of {column} for {str1} and {str2} are different. Since the reliability of the Mann Whitney U test is questionable when the distributions are different in the non-parametric case, we can only say that the distributions are different')
        else:
            print(f'We dont drop the null hypothesis and therefore retain that: The median/location of {column} for {str1} and {str2} are the same')

def perform_kw_test(df1, df2, df3, column, str1, str2, str3, df4 = None, str4 = None):
    # Conduct a Kruskal Wallis test to check if the distributions of AverageProfessorRating for the three groups are the same
    if(df4 is not None):
        kruskal_stat, kruskal_p_val = stats.kruskal(df1[column], df2[column], df3[column], df4[column])
    else:
        kruskal_stat, kruskal_p_val = stats.kruskal(df1[column], df2[column], df3[column])

    print('Kruskal Wallis Statistic: ', kruskal_stat)
    print('P-value: ', kruskal_p_val)
    if(kruskal_p_val < 0.005):
        print(f'We drop the null hypothesis and adopt: The distributions of {column} for ATLEAST ONE of {str1}, {str2} and {str3} are different')
    else:
        print(f'We dont drop the null hypothesis and therefore retain that: The distributions of {column} for {str1}, {str2} and {str3} are the same')
    

def perform_corr_test(df1, column1, column2, str1):
    print(f'Biserial Pearson Test of {str1} for the two groups: {column1} and {column2}')
    corr = stats.pointbiserialr(df1[column1], df1[column2])
    print('Biserial Pearson Correlation ', corr)

def visualize_95_ci(df, column, str1):
    # Calculate the 95% confidence interval for the sample means of male and female professors with more than 19 ratings and no pepper rating
    mean = df['AverageProfessorRating'].mean()
    std = df['AverageProfessorRating'].std()
    n = len(df['AverageProfessorRating'])

    ci_lower = mean - 1.96 * (std / np.sqrt(n))
    ci_upper = mean + 1.96 * (std / np.sqrt(n))

    print(f'95% Confidence Interval for Female Professors: [{ci_lower}, {ci_upper}]')

    # Plot the 95% confidence interval
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AverageProfessorRating'], bins=30, kde=True, color='blue', label=f'{column}')
    plt.axvline(ci_lower, color='red', linestyle='--', label=f'Lower CI: {ci_lower:.2f}')
    plt.axvline(df['AverageProfessorRating'].mean(), color='black', linestyle='-', label=f'Mean: {df["AverageProfessorRating"].mean():.2f}')
    plt.axvline(ci_upper, color='green', linestyle='--', label=f'Upper CI: {ci_upper:.2f}')
    plt.title(f'95% Confidence Interval for {column} of {str1}')
    plt.xlabel('AverageProfessorRating')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def effect_size(df1, df2, column):
    mean_diff = df1[column].mean() - df2[column].mean()

    pooledd_std = np.sqrt((df1[column].std()**2 + df2[column].std()**2) / 2)

    effect_size = mean_diff / pooledd_std

    print('Effect Size: ', effect_size)

def calculate_effect_size(data1, data2):
    mean_diff = np.mean(data1) - np.mean(data2)
    pooled_std = np.sqrt((np.std(data1, ddof=1) ** 2 + np.std(data2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std

def bootstrap_effect_size(data1, data2, num_bootstrap=1000, ci=95):
# Bootstrap method to calculate confidence intervals
    bootstrapped_effect_sizes = []
    for _ in range(num_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrapped_effect_sizes.append(calculate_effect_size(sample1, sample2))
    
    lower_bound = np.percentile(bootstrapped_effect_sizes, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_effect_sizes, 100 - (100 - ci) / 2)
    
    return lower_bound, upper_bound, bootstrapped_effect_sizes

def visualize_95_ci_effect_size(df1, df2, column, str1, str2):
    # Calculate the 95% confidence interval for the effect size
    lower_bound, upper_bound, bootstrapped_effect_sizes = bootstrap_effect_size(df1[column], df2[column])

    print(f'95% Confidence Interval for the Effect Size: [{lower_bound}, {upper_bound}]')

    # Plot the bootstrap distribution of effect sizes
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrapped_effect_sizes, bins=30, kde=True)
    plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound: {lower_bound:.3f}')
    plt.axvline(np.mean(bootstrapped_effect_sizes), color='black', linestyle='-', label=f'Mean: {np.mean(bootstrapped_effect_sizes):.3f}')
    plt.axvline(upper_bound, color='green', linestyle='--', label=f'Upper Bound: {upper_bound:.3f}')
    plt.title(f'Bootstrap Distribution of Effect Sizes for {str1} and {str2} for {column}')
    plt.xlabel('Effect Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def create_p_vals_df(df1):
    # For each of the tags, calculate the p-value of the gender bias using Mann-Whitney U test and the KS test and store the results in a dataframe
    # Initialize an empty list to store the results
    results = []

    # Iterate over each tag column
    for tag in df1.columns[5:]:
        male_values = df1[df1['HighConfMale'] == 1][tag].dropna()
        female_values = df1[df1['HighConfFemale'] == 1][tag].dropna()
        
        # Perform Mann-Whitney U test
        u_stat, p_value_u = stats.mannwhitneyu(male_values, female_values, alternative='two-sided')
        
        # Perform KS test
        ks_stat, p_value_ks = stats.ks_2samp(male_values, female_values)
        
        # Append the results to the list
        results.append({'Tag': tag, 'Mann-Whitney U p-value': p_value_u, 'KS test p-value': p_value_ks})

    # Convert the results list to a DataFrame
    p_values_df = pd.DataFrame(results)

    return p_values_df

def visualize_p_vals(p_vals_df, str1):
    # Plot the p-values of the Mann-Whitney U test and KS test for each tag
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title(f'P-values of Mann-Whitney U Test vs. KS Test for Each Tag for {str1}')
    sns.scatterplot(data=p_vals_df, x='Mann-Whitney U p-value', y='KS test p-value', ax=ax)
    for i, row in p_vals_df.iterrows():
        ax.text(row['Mann-Whitney U p-value'], row['KS test p-value'], row['Tag'], fontsize=7, rotation=60)
    ax.set_title('P-values of Mann-Whitney U Test vs. KS Test for Each Tag')
    ax.set_xlabel('Mann-Whitney U p-value')
    ax.set_ylabel('KS test p-value')
    plt.show()

def print_pvals(p_vals_df):

    significant_results = p_vals_df[(p_vals_df['Mann-Whitney U p-value'] < 0.05) | (p_vals_df['KS test p-value'] < 0.05)]
    # Get the 3 tags with the lowest p-values
    significant_results_smallest = p_vals_df.nsmallest(3, 'Mann-Whitney U p-value')

    # Additionally get the 3 tags with the lowest p-values and the 3 tags with the highest p-values
    significant_results_biggest = p_vals_df.nlargest(3, 'Mann-Whitney U p-value')

    print(significant_results)
    print(significant_results_smallest)
    print(significant_results_biggest)

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def normal_regression(X_train, y_train):
    return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

def ridge_regression(X_train, y_train, alpha):
    n_features = X_train.shape[1]
    identity = np.eye(n_features)
    identity[0, 0] = 0  
    return np.linalg.inv(X_train.T @ X_train + alpha * identity) @ X_train.T @ y_train

def lasso_regression(X_train, y_train, alpha, max_iter=10000, tol=1e-6):
    m, n = X_train.shape
    beta = np.zeros(n)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(n):
            residual = y_train - X_train @ beta + X_train[:, j] * beta[j]
            rho = X_train[:, j].T @ residual
            if j == 0:  
                beta[j] = rho / (X_train[:, j].T @ X_train[:, j])
            else:
                beta[j] = np.sign(rho) * max(0, abs(rho) - alpha) / (X_train[:, j].T @ X_train[:, j])
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

def getfinalresults(X_train,y_train,X_test,y_test,alphas=np.array([0.00001,0.0001,0.001,0.01,0.1,1,2,5,10,20,100,1000,2000,100000])):
    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
    X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

    # Normal Regression
    beta_normal = normal_regression(X_train_scaled, y_train)
    y_pred_normal = X_test_scaled @ beta_normal
    rmse_normal = compute_rmse(y_test, y_pred_normal)
    r2_normal = compute_r2(y_test, y_pred_normal)

    residuals = y_test - y_pred_normal

    ## Residual plot
    plt.figure(figsize=(10, 6))
    # Grab standardized residuals for plot
    std_residuals = (residuals - residuals.mean()) / residuals.std()
    plt.scatter(y_pred_normal, std_residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=0

    # Labels and title
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values (y_hat)")
    plt.ylabel("Standardized Residuals")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ## Pretty even spread in the residuals (homoscedasticity? - check)
    fig, ax = plt.subplots(figsize=(10,6))

    # See histogram of residuals to check for normality
    ax.hist(residuals, bins=15, color='green', edgecolor='black', density=True) # Add density
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    
    # Ridge Regression
    for alpha in alphas:
        beta_ridge = ridge_regression(X_train_scaled, y_train, alpha)
        y_pred_ridge = X_test_scaled @ beta_ridge
        rmse_ridge = compute_rmse(y_test, y_pred_ridge)
        r2_ridge = compute_r2(y_test, y_pred_ridge)
        results.append(('Ridge', alpha, rmse_ridge, r2_ridge))
    
    # Lasso Regression
    for alpha in alphas:
        beta_lasso = lasso_regression(X_train_scaled, y_train, alpha)
        y_pred_lasso = X_test_scaled @ beta_lasso
        rmse_lasso = compute_rmse(y_test, y_pred_lasso)
        r2_lasso = compute_r2(y_test, y_pred_lasso)
        results.append(('Lasso', alpha, rmse_lasso, r2_lasso))
        residuals = y_test - y_pred_lasso

        ## Residual plot
        plt.figure(figsize=(10, 6))
        # Grab standardized residuals for plot
        std_residuals = (residuals - residuals.mean()) / residuals.std()
        plt.scatter(y_pred_lasso, std_residuals, color='blue')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=0

        # Labels and title
        plt.title(f"Residual Plot lasso alpha= {alpha}")
        plt.xlabel("Predicted Values (y_hat)")
        plt.ylabel("Standardized Residuals")

        plt.grid(True)
        plt.tight_layout()
        plt.show()

        ## Pretty even spread in the residuals (homoscedasticity? - check)
        fig, ax = plt.subplots(figsize=(10,6))

        # See histogram of residuals to check for normality
        ax.hist(residuals, bins=15, color='green', edgecolor='black', density=True) # Add density
        ax.set_title("Histogram of Residuals")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
    
    results.append(('Normal', None, rmse_normal, r2_normal))
    return results
def getKFresults(X,y,alphas=np.array([0.00001,0.0001,0.001,0.01,0.1,1,2,5,10,20,100,1000,2000,100000])):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
        X_val_scaled = np.hstack((np.ones((X_val_scaled.shape[0], 1)), X_val_scaled))

        # Normal Regression
        beta_normal = normal_regression(X_train_scaled, y_train)
        y_pred_normal = X_val_scaled @ beta_normal
        rmse_normal = compute_rmse(y_val, y_pred_normal)
        r2_normal = compute_r2(y_val, y_pred_normal)
        
        # Ridge Regression
        for alpha in alphas:
            beta_ridge = ridge_regression(X_train_scaled, y_train, alpha)
            y_pred_ridge = X_val_scaled @ beta_ridge
            rmse_ridge = compute_rmse(y_val, y_pred_ridge)
            r2_ridge = compute_r2(y_val, y_pred_ridge)
            results.append(('Ridge', beta_ridge,alpha, rmse_ridge, r2_ridge))
        
        # Lasso Regression
        for alpha in alphas:
            beta_lasso = lasso_regression(X_train_scaled, y_train, alpha)
            y_pred_lasso = X_val_scaled @ beta_lasso
            rmse_lasso = compute_rmse(y_val, y_pred_lasso)
            r2_lasso = compute_r2(y_val, y_pred_lasso)
            results.append(('Lasso', beta_lasso, alpha, rmse_lasso, r2_lasso))
        
        results.append(('Normal',beta_normal, None, rmse_normal, r2_normal))
    return results

def getbetas(X_train,y_train,X_test,y_test):
    results=[]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
    X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

    # Normal Regression
    beta_normal = normal_regression(X_train_scaled, y_train)
    y_pred_normal = X_test_scaled @ beta_normal
    rmse_normal = compute_rmse(y_test, y_pred_normal)
    r2_normal = compute_r2(y_test, y_pred_normal)
    results.append(('Normal',beta_normal, rmse_normal, r2_normal))
    return beta_normal,rmse_normal,r2_normal

def getbetaslasso(X_train,y_train,X_test,y_test,alpha=10):
    results=[]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
    X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

    # Normal Regression
    beta_lasso = lasso_regression(X_train_scaled, y_train, alpha)
    y_pred_lasso = X_test_scaled @ beta_lasso
    rmse_lasso = compute_rmse(y_test, y_pred_lasso)
    r2_lasso = compute_r2(y_test, y_pred_lasso)
    results.append(('Lasso', beta_lasso, rmse_lasso, r2_lasso))
    return beta_lasso,rmse_lasso,r2_lasso

def getbetasridge(X_train,y_train,X_test,y_test,alpha=10):
    results=[]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
    X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

    # Normal Regression
    beta_ridge = ridge_regression(X_train_scaled, y_train, alpha)
    y_pred_ridge = X_test_scaled @ beta_ridge
    rmse_ridge = compute_rmse(y_test, y_pred_ridge)
    r2_ridge = compute_r2(y_test, y_pred_ridge)
    results.append(('ridge', beta_ridge, rmse_ridge, r2_ridge))
    return beta_ridge,rmse_ridge,r2_ridge

def plot_results(results_df,alphas=np.array([0.00001,0.0001,0.001,0.01,0.1,1,2,5,10,20,100,1000,2000,100000])):
    ridge_results = results_df[results_df['Model'] == 'Ridge'].drop(columns=['Model']).groupby('Alpha').mean()
    lasso_results = results_df[results_df['Model'] == 'Lasso'].drop(columns=['Model']).groupby('Alpha').mean()
    normal_results = results_df[results_df['Model']=='Normal'].drop(columns=['Model','Alpha']).mean()
    
    plt.figure(figsize=(10, 6))
    plt.xscale("log")

    plt.plot(alphas, ridge_results['RMSE'], marker='o', label='Ridge Regression RMSE')
    plt.plot(alphas, lasso_results['RMSE'], marker='s', label='Lasso Regression RMSE')
    
    plt.axhline(y=normal_results['RMSE'], color='r', linestyle='--', label='Normal Regression RMSE')

    plt.title('RMSE Comparison Across Models')
    plt.xlabel('Alpha (Regularization Parameter)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.xscale("log")

    plt.plot(alphas, ridge_results['R2'], marker='o', label='Ridge Regression R2')
    plt.plot(alphas, lasso_results['R2'], marker='s', label='Lasso Regression R2')

    plt.axhline(y=normal_results['R2'], color='r', linestyle='--', label='Normal Regression R2')

    plt.title('R2 Comparison Across Models')
    plt.xlabel('Alpha (Regularization Parameter)')
    plt.ylabel('R2')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_betas(betas,feature_names=0,mytype='Normal',alpha=10):
    # Plot the coefficients (betas) for all models
    plt.figure(figsize=(24, 6))
    # Ridge Regression Coefficients (for best alpha)
    if(feature_names==0):
        feature_names=range(len(betas))
    #plt.xticks(range(len(betas)))
    plt.bar([str(i) for i in feature_names], betas)
    if mytype=='Normal':
        plt.title(f'Coefficients Regression)')
    if mytype=='Ridge':
        plt.title(f'Coefficients Ridge (alpha={alpha})')
    if mytype=='Lasso':
        plt.title(f'Coefficients Ridge (alpha={alpha})')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')

    plt.tight_layout()
    plt.show()
def scatterdepandindependent(df,dependent_var):
    for var in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[var], y=df[dependent_var])
        plt.title(f'Scatterplot of {dependent_var} vs {var}')
        plt.xlabel(var)
        plt.ylabel(dependent_var)
        plt.grid(True)
        plt.show()
def plot_forward_selection_results(results_df):
    plt.figure(figsize=(12, 6))

    # RMSE plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(results_df) + 1), results_df['RMSE'], marker='o', label='RMSE')
    plt.title('Forward Feature Selection: RMSE')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('RMSE')
    plt.grid(True)

    # R² plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(results_df) + 1), results_df['R2'], marker='o', label='R²')
    plt.title('Forward Feature Selection: R²')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('R²')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
def forward_feature_selection_kfold(X, y, k=5, max_features=None):
    """
    Perform forward feature selection using k-fold cross-validation.

    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - k: Number of folds for cross-validation
    - max_features: Max number of features to select. If None, select all.

    Returns:
    - selected_features: List of selected features
    - results: List of results for each feature set (including average RMSE, R², etc.)
    """
    remaining_features = list(X.columns)
    selected_features = []
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    while remaining_features:
        best_rmse = float('inf')
        best_feature = None
        best_betas = None
        best_alpha = None
        
        for feature in remaining_features:
            # Create a subset of the data with the current selected features + one more
            X_temp = X[selected_features + [feature]]
            
            rmse_list = []
            r2_list = []
            
            for train_idx, val_idx in kf.split(X_temp):
                X_train_fold, X_val_fold = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train a model
                model = LinearRegression()
                model.fit(X_train_fold, y_train_fold)
                
                # Make predictions and evaluate the model
                y_pred_fold = model.predict(X_val_fold)
                rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                r2_fold = r2_score(y_val_fold, y_pred_fold)
                
                rmse_list.append(rmse_fold)
                r2_list.append(r2_fold)
            
            # Calculate average RMSE and R² across folds
            avg_rmse = np.mean(rmse_list)
            avg_r2 = np.mean(r2_list)
            
            # Store the best feature and corresponding metrics
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_feature = feature
                best_betas = model.coef_
                best_alpha = model.intercept_
        
        # Update the list of selected features
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        # Append the result for the current model
        results.append([selected_features.copy(), best_betas, best_alpha, best_rmse, avg_r2])
        
        # Stop if we've reached the maximum number of features (if specified)
        if max_features is not None and len(selected_features) >= max_features:
            break

    return selected_features, results

def main():
    print("Question 1")
    df_capstone = pd.read_csv('./rmpCapstoneNum.csv', header=0)
    tagsdf=pd.read_csv('./rmpCapstoneTags.csv', header=None)
    df_capstone.columns = ['AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper', 
                       'Proportion of students that said they would take the class again', 
                       'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale']
    df_capstone_greater_than_10_all = df_capstone[(df_capstone['NumberOfRatings'] >= 10)]
    df_capstone_greater_than_10 = df_capstone[(df_capstone['NumberOfRatings'] >= 10) & ~((df_capstone['HighConfMale'] == 1) & (df_capstone['HighConfFemale'] == 1)) & ~((df_capstone['HighConfMale'] == 0) & (df_capstone['HighConfFemale'] == 0))]
    df_capstone_greater_than_10_men = df_capstone_greater_than_10[df_capstone_greater_than_10['HighConfMale'] == 1]
    df_capstone_greater_than_10_female = df_capstone_greater_than_10[df_capstone_greater_than_10['HighConfFemale'] == 1]
    print("For Average Professor Rating with professors with more than 10 ratings")
    print("------------------------------------")
    visualize_density_plot(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating', 'df_capstone_greater_than_10_men', 'df_capstone_greater_than_10_female')
    perform_ks_mw_test(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating', "df_capstone_greater_than_10_men", "df_capstone_greater_than_10_female")
    print("------------------------------------")
    print("Adjusting for confounding variables")
    print("Check for pepper")
    print("------------------------------------")
    visualize_density_plot(df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 1], df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 0], 'AverageProfessorRating', 'df_capstone_greater_than_10[Received a pepper] == 1', 'df_capstone_greater_than_10[Received a pepper] == 0')
    perform_corr_test(df_capstone_greater_than_10, 'Received a pepper', 'AverageProfessorRating', "df_capstone_greater_than_10")
    print("------------")
    perform_ks_mw_test(df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 1], df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 0], 'AverageProfessorRating', "Received a pepper", "Did not receive a pepper")
    print("------------------------------------")
    print("Check if years of experience is a confounding variable")
    print("------------------------------------")
    df_capstone_10_to_12 = df_capstone_greater_than_10[(df_capstone_greater_than_10['NumberOfRatings'] >= 10) & (df_capstone_greater_than_10['NumberOfRatings'] <= 12)]
    df_capstone_13_to_18 = df_capstone_greater_than_10[(df_capstone_greater_than_10['NumberOfRatings'] >= 13) & (df_capstone_greater_than_10['NumberOfRatings'] <= 18)]
    df_capstone_19_plus = df_capstone_greater_than_10[(df_capstone_greater_than_10['NumberOfRatings'] > 18)]
    visualize_density_plot(df_capstone_10_to_12, df_capstone_13_to_18, 'AverageProfessorRating', 'df_capstone_10_to_12', 'df_capstone_13_to_18', df_capstone_19_plus, 'df_capstone_19_plus')
    perform_corr_test(df_capstone_greater_than_10, 'NumberOfRatings', 'AverageProfessorRating', "df_capstone_greater_than_10")
    perform_kw_test(df_capstone_10_to_12, df_capstone_13_to_18, df_capstone_19_plus, 'AverageProfessorRating', 'df_capstone_10_to_12', 'df_capstone_13_to_18', 'df_capstone_19_plus')
    perform_ks_mw_test(df_capstone_10_to_12, df_capstone_13_to_18, 'AverageProfessorRating', "df_capstone_10_to_12", "df_capstone_13_to_18")
    perform_ks_mw_test(df_capstone_13_to_18, df_capstone_19_plus, 'AverageProfessorRating', "df_capstone_13_to_18", "df_capstone_19_plus")
    perform_ks_mw_test(df_capstone_10_to_12, df_capstone_19_plus, 'AverageProfessorRating', "df_capstone_10_to_12", "df_capstone_19_plus")
    print("------------------------------------")
    df_capstone_10_to_18 = df_capstone_greater_than_10[(df_capstone_greater_than_10['NumberOfRatings'] >= 10) & (df_capstone_greater_than_10['NumberOfRatings'] <= 18)]
    visualize_density_plot(df_capstone_10_to_18, df_capstone_19_plus, 'AverageProfessorRating', 'df_capstone_10_to_18', 'df_capstone_19_plus')
    perform_ks_mw_test(df_capstone_10_to_18, df_capstone_19_plus, 'AverageProfessorRating', "df_capstone_10_to_18", "df_capstone_19_plus")
    print("------------------------------------")
    print("We adjust for the confounding varaibles: Pepper and Years of Experience")
    df_capstone_10_to_18_pepper = df_capstone_10_to_18[df_capstone_10_to_18['Received a pepper'] == 1]
    df_capstone_10_to_18_no_pepper = df_capstone_10_to_18[df_capstone_10_to_18['Received a pepper'] == 0]
    df_capstone_19_plus_pepper = df_capstone_19_plus[df_capstone_19_plus['Received a pepper'] == 1]
    df_capstone_19_plus_no_pepper = df_capstone_19_plus[df_capstone_19_plus['Received a pepper'] == 0]
    # Pepper
    df_capstone_10_to_18_pepper_male = df_capstone_10_to_18_pepper[df_capstone_10_to_18_pepper['HighConfMale'] == 1]
    df_capstone_10_to_18_pepper_female = df_capstone_10_to_18_pepper[df_capstone_10_to_18_pepper['HighConfFemale'] == 1]
    df_capstone_19_plus_pepper_male = df_capstone_19_plus_pepper[df_capstone_19_plus_pepper['HighConfMale'] == 1]
    df_capstone_19_plus_pepper_female = df_capstone_19_plus_pepper[df_capstone_19_plus_pepper['HighConfFemale'] == 1]

    # No Pepper
    df_capstone_10_to_18_no_pepper_male = df_capstone_10_to_18_no_pepper[df_capstone_10_to_18_no_pepper['HighConfMale'] == 1]
    df_capstone_10_to_18_no_pepper_female = df_capstone_10_to_18_no_pepper[df_capstone_10_to_18_no_pepper['HighConfFemale'] == 1]
    df_capstone_19_plus_no_pepper_male = df_capstone_19_plus_no_pepper[df_capstone_19_plus_no_pepper['HighConfMale'] == 1]
    df_capstone_19_plus_no_pepper_female = df_capstone_19_plus_no_pepper[df_capstone_19_plus_no_pepper['HighConfFemale'] == 1]

    perform_ks_mw_test(df_capstone_10_to_18_pepper_male, df_capstone_10_to_18_pepper_female, 'AverageProfessorRating', "10_to_18_pepper_male", "10_to_18_pepper_female")
    perform_ks_mw_test(df_capstone_10_to_18_no_pepper_male, df_capstone_10_to_18_no_pepper_female, 'AverageProfessorRating', "10_to_18_no_pepper_male", "10_to_18_no_pepper_female")
    perform_ks_mw_test(df_capstone_19_plus_pepper_male, df_capstone_19_plus_pepper_female, 'AverageProfessorRating', "19_plus_pepper_male", "19_plus_pepper_female")
    perform_ks_mw_test(df_capstone_19_plus_no_pepper_male, df_capstone_19_plus_no_pepper_female, 'AverageProfessorRating', "19_plus_no_pepper_male", "19_plus_no_pepper_female")
    print("------------------------------------")
    print("We check the only significant result: 19_plus_no_pepper")
    visualize_density_plot(df_capstone_19_plus_no_pepper_male, df_capstone_19_plus_no_pepper_female, 'AverageProfessorRating', '19_plus_no_pepper_male', '19_plus_no_pepper_female')
    visualize_95_ci(df_capstone_19_plus_no_pepper_male, 'AverageProfessorRating', '19_plus_no_pepper_male')
    visualize_95_ci(df_capstone_19_plus_no_pepper_female, 'AverageProfessorRating', '19_plus_no_pepper_female')
    print("------------------------------------")
    print("Question 3")
    print("Lets check the effect sizes: Cohen's D")
    print("------------------------------------")
    print("For 19_plus_no_pepper")
    effect_size(df_capstone_19_plus_no_pepper_male, df_capstone_19_plus_no_pepper_female, 'AverageProfessorRating')
    visualize_95_ci_effect_size(df_capstone_19_plus_no_pepper_male, df_capstone_19_plus_no_pepper_female, 'AverageProfessorRating', '19_plus_no_pepper_male', '19_plus_no_pepper_female')
    print("------------------------------------")
    print('For professors with 10 or more ratings')
    effect_size(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating')
    visualize_95_ci_effect_size(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating', 'df_capstone_greater_than_10_male', 'df_capstone_greater_than_10_female')
    print("------------------------------------")
    print("Question 4")
    df_capstone_tags = pd.read_csv('./rmpCapstoneTags.csv', header=None)
    df_capstone_tags.columns = ['Tough grader', 'Good feedback', 'Respected', 'Lots to read', 'Participation matters', 
                                'Don’t skip class or you will not pass', 'Lots of homework', 'Inspirational', 'Pop quizzes!', 
                                'Accessible', 'So many papers', 'Clear grading', 'Hilarious', 'Test heavy', 'Graded by few things', 
                                'Amazing lectures', 'Caring', 'Extra credit', 'Group projects', 'Lecture heavy']
    # Merge the two dataframes
    df_merged = pd.concat([df_capstone[['AverageProfessorRating', 'NumberOfRatings', 'Received a pepper', 'HighConfMale', 'HighConfFemale']], df_capstone_tags], axis=1)
    df_merged.head()

    df_merged_min_10 =  df_merged[(df_merged['NumberOfRatings'] >= 10) & ~((df_merged['HighConfMale'] == 1) & (df_merged['HighConfFemale'] == 1)) & ~((df_merged['HighConfMale'] == 0) & (df_merged['HighConfFemale'] == 0))]
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    df_merged_min_10.iloc[:, 5:] = df_merged_min_10.iloc[:, 5:].astype(float)

    print("For professors with more than 10 ratings")
    print("------------------------------------")
    # Replace tag values by normalizing them with the total number of tags awarded to that professor
    df_merged_min_10.iloc[:, 5:] = df_merged_min_10.iloc[:, 5:].div(df_merged_min_10.iloc[:, 5:].sum(axis=1), axis=0)

    p_vals_min_10 = create_p_vals_df(df_merged_min_10)
    visualize_p_vals(p_vals_min_10, 'For professors with more than 10 ratings')
    print_pvals(p_vals_min_10)
    print("------------------------------------")
    print("For professors with more than 19 ratings and no pepper rating")
    df_merged_min_19_no_pepper = df_merged_min_10[(df_merged_min_10['Received a pepper'] == 0) & (df_merged_min_10['NumberOfRatings'] >= 19)]
    p_vals_min_19_no_pepper = create_p_vals_df(df_merged_min_19_no_pepper)
    visualize_p_vals(p_vals_min_19_no_pepper, 'For professors with more than 19 ratings and no pepper rating')
    print_pvals(p_vals_min_19_no_pepper)
    visualize_density_plot(df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfMale'] == 1], df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfFemale'] == 1], 'Pop quizzes!', 'HighConfMale', 'HighConfFemale', nbins = 3)
    print("Median: df_merged_min_19_no_pepper male", df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfMale'] == 1]['Pop quizzes!'].describe(), "Median: df_merged_min_19_no_pepper female", df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfFemale'] == 1]['Pop quizzes!'].describe())
    print("------------------------------------")
    
    print("Question 7")
    print("Comparing NA vs non NA distributions to help decide if rows should be dropped")
    df_capstone_na_dropped=df_capstone_greater_than_10_all.dropna()
    print(df_capstone_na_dropped['AverageProfessorRating'].mean(),df_capstone_na_dropped['AverageProfessorRating'].median(),df_capstone_na_dropped['AverageProfessorRating'].std())
    df_only_na=df_capstone_greater_than_10_all[df_capstone_greater_than_10_all.isnull().any(axis=1)]
    print(df_only_na['AverageProfessorRating'].mean(),df_only_na['AverageProfessorRating'].median(),df_only_na['AverageProfessorRating'].std())
    sns.boxplot(df_only_na['AverageProfessorRating'])
    plt.title('Distribution of AverageProfessorRating if Proportion column is missing')
    plt.show()
    sns.boxplot(df_capstone_na_dropped['AverageProfessorRating'])
    plt.title('Distribution of AverageProfessorRating if Proportion column is present')
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.histplot(df_capstone_na_dropped['AverageProfessorRating'], bins=30, kde=True, color='blue', label='AverageProfessorRating if prop not missing', stat='density')
    sns.histplot(df_only_na['AverageProfessorRating'], bins=30, kde=True, color='red', label='AverageProfessorRating if prop missing', stat='density')
    plt.title('')
    plt.xlabel('AverageProfessorRating')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    df_capstone_na_dropped['Proportion of online class ratings'] = df_capstone_na_dropped['Number of ratings coming from online classes'].div(df_capstone_na_dropped['NumberOfRatings'])
    
    correlation_matrix = df_capstone_na_dropped.corr()

    sns.heatmap(correlation_matrix,cmap = "RdBu_r", annot=True)
    plt.title('Correlation Matrix')
    plt.show()
    df_capstone_dropped_final=df_capstone_na_dropped
    df_capstone_dropped_final=df_capstone_dropped_final[(df_capstone_dropped_final['HighConfMale']==1) & (df_capstone_dropped_final['HighConfFemale']==0) | (df_capstone_dropped_final['HighConfMale']==0) & (df_capstone_dropped_final['HighConfFemale']==1)]
    dependent_var='AverageProfessorRating'
    scatterdepandindependent(df_capstone_dropped_final,dependent_var)

    X = np.array([
    df_capstone_dropped_final['Average Difficulty'],
    df_capstone_dropped_final['NumberOfRatings'],
    df_capstone_dropped_final['Received a pepper'],
    df_capstone_dropped_final['HighConfFemale'],
    df_capstone_dropped_final['Proportion of online class ratings'],
    df_capstone_dropped_final['Proportion of students that said they would take the class again']
    ]).T

    y = np.array(df_capstone_dropped_final['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X,y,alphas=np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000,1500, 100000]))

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')

    plot_results(results_df,alphas=np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000,1500, 100000]))

    results=getfinalresults(X_train,y_train,X_test,y_test,alphas=np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000,2000, 100000]))
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df,alphas=np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000,2000, 100000]))
    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)  
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:],[i.name for i in [
        df_capstone_dropped_final['Average Difficulty'],
        df_capstone_dropped_final['NumberOfRatings'],
        df_capstone_dropped_final['Received a pepper'],
        df_capstone_dropped_final['HighConfFemale'],
        df_capstone_dropped_final['Proportion of online class ratings'],
        df_capstone_dropped_final['Proportion of students that said they would take the class again']
    ]])
    X = np.array([
    df_capstone_dropped_final['Proportion of students that said they would take the class again'],
    df_capstone_dropped_final['Average Difficulty'],
    df_capstone_dropped_final['Received a pepper']
    ]).T

    y = np.array(df_capstone_dropped_final['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
    print(indices)  
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:],[i.name for i in [
        df_capstone_dropped_final['Proportion of students that said they would take the class again'],
        df_capstone_dropped_final['Average Difficulty'],
        df_capstone_dropped_final['Received a pepper']
    ]])
    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)
    X = np.array([
    df_capstone_dropped_final['Proportion of students that said they would take the class again']
    ]).T

    y = np.array(df_capstone_dropped_final['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model', 'Betas','Alpha', 'RMSE', 'R2'])
    results_df=results_df2.drop(columns=['Betas'])
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
    print(indices)  
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:])
    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)
    print("Question 8")
    Q8df=df_capstone[['AverageProfessorRating','NumberOfRatings']].join(tagsdf, how='inner')
    Q8df=df_capstone[['AverageProfessorRating','NumberOfRatings']].join(tagsdf, how='inner')
    Q8df.dropna(inplace=True)
    for i in Q8df.columns[2:]:
        Q8df[i] = Q8df[i].div(Q8df['NumberOfRatings'])
    Q8dfgreater10=Q8df[Q8df['NumberOfRatings'] >= 10]
    correlation_matrix = Q8dfgreater10.drop(columns=['NumberOfRatings','AverageProfessorRating']).corr()
    plt.figure(figsize = (40,40))
    sns.heatmap(correlation_matrix,cmap = "RdBu_r", annot=True)
    plt.title('Correlation Matrix')
    plt.show()
    dependent_var='AverageProfessorRating'
    scatterdepandindependent(Q8dfgreater10,dependent_var)
    X = Q8dfgreater10.drop(columns=['AverageProfessorRating','NumberOfRatings'])  # assuming all columns except 'AverageProfessorRating' are features
    y = Q8dfgreater10['AverageProfessorRating']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Perform forward feature selection with k-fold cross-validation
    selected_features_kfold, forward_results_kfold = forward_feature_selection_kfold(X_train, y_train, k=5)

    # Convert results to a DataFrame
    forward_results_kfold_df = pd.DataFrame(forward_results_kfold, columns=['Selected Features', 'Betas', 'Alpha', 'RMSE', 'R2'])

    # Optionally, remove the 'Betas' column for cleaner display
    forward_results_kfold_df_cleaned = forward_results_kfold_df.drop(columns='Betas')

    # Plot the results
    plot_forward_selection_results(forward_results_kfold_df_cleaned)

    # Evaluate the final model on the test set
    best_selected_features_kfold = forward_results_kfold_df.iloc[-1]['Selected Features']
    X_train_best_kfold = X_train[best_selected_features_kfold]
    X_test_best_kfold = X_test[best_selected_features_kfold]

    final_model_kfold = LinearRegression()
    final_model_kfold.fit(X_train_best_kfold, y_train)
    y_pred_test_kfold = final_model_kfold.predict(X_test_best_kfold)

    final_rmse_kfold = np.sqrt(mean_squared_error(y_test, y_pred_test_kfold))
    final_r2_kfold = r2_score(y_test, y_pred_test_kfold)

    print(f"Final Model with K-Fold RMSE: {final_rmse_kfold}")
    print(f"Final Model with K-Fold R²: {final_r2_kfold}")
    print(forward_results_kfold_df)

    X = np.array([
    Q8dfgreater10[0],
    Q8dfgreater10[1],
    Q8dfgreater10[2],
    Q8dfgreater10[3],
    Q8dfgreater10[4],
    Q8dfgreater10[5],
    Q8dfgreater10[6],
    Q8dfgreater10[7],
    Q8dfgreater10[8],
    Q8dfgreater10[9],
    Q8dfgreater10[10],
    Q8dfgreater10[11],
    Q8dfgreater10[12],
    Q8dfgreater10[13],
    Q8dfgreater10[14],
    Q8dfgreater10[15],
    Q8dfgreater10[16],
    Q8dfgreater10[17],
    Q8dfgreater10[18],
    Q8dfgreater10[19],
    ]).T

    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
    print(indices)  
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:],[i.name for i in [
        Q8dfgreater10[0],
        Q8dfgreater10[1],
        Q8dfgreater10[2],
        Q8dfgreater10[3],
        Q8dfgreater10[4],
        Q8dfgreater10[5],
        Q8dfgreater10[6],
        Q8dfgreater10[7],
        Q8dfgreater10[8],
        Q8dfgreater10[9],
        Q8dfgreater10[10],
        Q8dfgreater10[11],
        Q8dfgreater10[12],
        Q8dfgreater10[13],
        Q8dfgreater10[14],
        Q8dfgreater10[15],
        Q8dfgreater10[16],
        Q8dfgreater10[17],
        Q8dfgreater10[18],
        Q8dfgreater10[19],
    ]])
    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    X = np.array([
    Q8dfgreater10[2]
    ]).T
    X_sqrt = np.sqrt(X)   # Square root of the column
    X_square = np.square(X)  # Square of the column
    X_cube = np.power(X, 3)  # Cube of the column

    X_extended = np.hstack((X, X_sqrt, X_square, X_cube))
    X=X_extended
    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    X = np.array([
    Q8dfgreater10[0]
    ]).T

    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    print(getbetas(X_train,y_train,X_test,y_test))
    X = np.array([
    Q8dfgreater10[0],
    Q8dfgreater10[1],
    Q8dfgreater10[2],
    Q8dfgreater10[15],
    Q8dfgreater10[16],
    Q8dfgreater10[12],
    ]).T
    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:])
    print('Question 9')
    Q9df=df_capstone[['Average Difficulty','NumberOfRatings']].join(tagsdf, how='inner')

    Q9df.dropna(inplace=True)

    for i in Q9df.columns[2:]:
        Q9df[i] = Q9df[i].div(Q9df['NumberOfRatings'])

    Q9dfgreater10=Q9df[Q9df['NumberOfRatings'] >= 10]


    print(Q9dfgreater10.corr())
    dependent_var='Average Difficulty'
    scatterdepandindependent(Q9dfgreater10)
    X = Q9dfgreater10.drop(columns=['Average Difficulty','NumberOfRatings'])  # assuming all columns except 'AverageProfessorRating' are features
    y = Q9dfgreater10['Average Difficulty']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform forward feature selection with k-fold cross-validation
    selected_features_kfold, forward_results_kfold = forward_feature_selection_kfold(X_train, y_train, k=5)

    # Convert results to a DataFrame
    forward_results_kfold_df = pd.DataFrame(forward_results_kfold, columns=['Selected Features', 'Betas', 'Alpha', 'RMSE', 'R2'])

    # Optionally, remove the 'Betas' column for cleaner display
    forward_results_kfold_df_cleaned = forward_results_kfold_df.drop(columns='Betas')

    # Plot the results
    plot_forward_selection_results(forward_results_kfold_df_cleaned)

    # Evaluate the final model on the test set
    best_selected_features_kfold = forward_results_kfold_df.iloc[-1]['Selected Features']
    X_train_best_kfold = X_train[best_selected_features_kfold]
    X_test_best_kfold = X_test[best_selected_features_kfold]

    final_model_kfold = LinearRegression()
    final_model_kfold.fit(X_train_best_kfold, y_train)
    y_pred_test_kfold = final_model_kfold.predict(X_test_best_kfold)

    final_rmse_kfold = np.sqrt(mean_squared_error(y_test, y_pred_test_kfold))
    final_r2_kfold = r2_score(y_test, y_pred_test_kfold)

    print(f"Final Model with K-Fold RMSE: {final_rmse_kfold}")
    print(f"Final Model with K-Fold R²: {final_r2_kfold}")

    print(forward_results_kfold_df)

    X = np.array([
        Q9dfgreater10[0],
        Q9dfgreater10[1],
        Q9dfgreater10[2],
        Q9dfgreater10[3],
        Q9dfgreater10[4],
        Q9dfgreater10[5],
        Q9dfgreater10[6],
        Q9dfgreater10[7],
        Q9dfgreater10[8],
        Q9dfgreater10[9],
        Q9dfgreater10[10],
        Q9dfgreater10[11],
        Q9dfgreater10[12],
        Q9dfgreater10[13],
        Q9dfgreater10[14],
        Q9dfgreater10[15],
        Q9dfgreater10[16],
        Q9dfgreater10[17],
        Q9dfgreater10[18],
        Q9dfgreater10[19],
    ]).T

    y = np.array(Q9dfgreater10['Average Difficulty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetaslasso(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)
    print(mybetas[indices])
    print(getbetaslasso(X_train,y_train,X_test,y_test,alpha=1000))
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetaslasso(X_train,y_train,X_test,y_test,alpha=1000)[0][1:],[i.name for i in [
        Q9dfgreater10[0],
        Q9dfgreater10[1],
        Q9dfgreater10[2],
        Q9dfgreater10[3],
        Q9dfgreater10[4],
        Q9dfgreater10[5],
        Q9dfgreater10[6],
        Q9dfgreater10[7],
        Q9dfgreater10[8],
        Q9dfgreater10[9],
        Q9dfgreater10[10],
        Q9dfgreater10[11],
        Q9dfgreater10[12],
        Q9dfgreater10[13],
        Q9dfgreater10[14],
        Q9dfgreater10[15],
        Q9dfgreater10[16],
        Q9dfgreater10[17],
        Q9dfgreater10[18],
        Q9dfgreater10[19],
    ]],'Lasso')


    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    X = np.array([
        Q9dfgreater10[0]
    ]).T
    X_sqrt = np.sqrt(X)   # Square root of the column
    X_square = np.square(X)  # Square of the column
    X_cube = np.power(X, 3)  # Cube of the column

    X_extended = np.hstack((X, X_sqrt, X_square, X_cube))
    X=X_extended
    y = np.array(Q9dfgreater10['Average Difficulty'])
    #y = np.sqrt(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)


    X = np.array([
        Q9dfgreater10[0],
        Q9dfgreater10[13],
        Q9dfgreater10[6],
        Q9dfgreater10[3],
        Q9dfgreater10[9]
    ]).T

    X_sqrt = np.sqrt(X)   # Square root of the column

    X_extended = np.hstack((X, X_sqrt))
    X=X_extended
    y = np.array(Q9dfgreater10['Average Difficulty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    mybetas=getbetaslasso(X_train,y_train,X_test,y_test,alpha=1000)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)
    print(getbetaslasso(X_train,y_train,X_test,y_test,alpha=100))
    plot_betas(getbetaslasso(X_train,y_train,X_test,y_test,alpha=100)[0][1:])

    X = np.array([
        Q9dfgreater10[0],
        Q9dfgreater10[13],
        Q9dfgreater10[6],
        Q9dfgreater10[3],
        Q9dfgreater10[9]
    ]).T

    y = np.array(Q9dfgreater10['Average Difficulty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:],[i.name for i in [
        Q9dfgreater10[0],
        Q9dfgreater10[13],
        Q9dfgreater10[6],
        Q9dfgreater10[3],
        Q9dfgreater10[9]
    ]])

    X = np.array([
        Q9dfgreater10[0],
    ]).T

    y = np.array(Q9dfgreater10['Average Difficulty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    print(indices)
    print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:])

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)
            
if __name__ == '__main__':
    main()