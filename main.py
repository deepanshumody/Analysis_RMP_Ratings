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
from scipy.stats import levene
from math import sqrt
from scipy.stats import t
from scipy.stats import chi2_contingency
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report)
from sklearn.preprocessing import MinMaxScaler

# Seed value for random number generators to obtain reproducible results
RANDOM_SEED = 10676128

# Apply the random seed to numpy.
np.random.seed(RANDOM_SEED)

pd.options.mode.chained_assignment = None  # default='warn'


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
    print(f'Biserial Pearson Test of correlation {str1} for the two groups: {column1} and {column2}')
    corr = stats.pointbiserialr(df1[column1], df1[column2])
    print('Biserial Pearson Correlation ', corr)

def perform_corr_cont_test(df1, column1, column2, str1):
    print(f'Pearson Test of correlation {str1} for the two groups: {column1} and {column2}')
    corr = df1[column1].corr(df1[column2])
    print('Pearson Correlation ', corr)

def visualize_95_ci(df, column, str1):
    # Calculate the 95% confidence interval for the sample means of male and female professors with more than 19 ratings and no pepper rating
    mean = df['AverageProfessorRating'].mean()
    std = df['AverageProfessorRating'].std()
    n = len(df['AverageProfessorRating'])

    ci_lower = mean - 1.96 * (std / np.sqrt(n))
    ci_upper = mean + 1.96 * (std / np.sqrt(n))

    print(f'95% Confidence Interval for {str1} Professors: [{ci_lower}, {ci_upper}]')

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
def lavenes_test(df1, df2, column):
    # Extract Average Ratings for Male and Female Professors
    ratings_male = df1[column]
    ratings_female = df2[column]

    # Perform Levene's Test
    stat, p_value = levene(ratings_male, ratings_female)

    # Display the results
    print(f"Levene's Test Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    if p_value < 0.005:  # Using significance level of 0.005
        print("The variances are significantly different (reject the null hypothesis).")
    else:
        print("The variances are not significantly different (fail to reject the null hypothesis).")

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
def lavenes_test_group(group_distributions):
    stat, p_value = levene(*group_distributions)  # Unpack the list of distributions
    print(f"Levene's Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")

    # Interpretation
    if p_value < 0.005:
        print("The variances are significantly different (reject the null hypothesis).")
    else:
        print("The variances are not significantly different (fail to reject the null hypothesis).")

# Function to calculate pooled standard deviation
def pooled_std(group1, group2):
    n1, n2 = len(group1), len(group2)
    return sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))

# Function to calculate confidence interval for Cohen's d
def cohen_d_confidence_interval(group1, group2, column, alpha=0.005):
    group1 = group1[column]
    group2 = group2[column]
    n1, n2 = len(group1), len(group2)
    d = (np.mean(group1) - np.mean(group2)) / pooled_std(group1, group2)
    se_d = sqrt((n1 + n2) / (n1 * n2) + (d**2 / (2 * (n1 + n2))))
    dof = n1 + n2 - 2
    t_crit = t.ppf(1 - alpha / 2, dof)
    margin_of_error = t_crit * se_d
    lower = d - margin_of_error
    upper = d + margin_of_error
    print(f"Cohen's d (Effect Size): {d:.3f}")
    print(f"95% Confidence Interval for Cohen's d: ({lower:.3f}, {upper:.3f})") 
    return d, lower, upper

def create_num_ratings_group(df):
    # Step 1: Create Two Groups (Half and Half) for Number of Ratings
    warnings.filterwarnings("ignore", category=FutureWarning)
    df.loc[:, 'Ratings Group'] = pd.qcut(
        df['NumberOfRatings'], 
        q=2,  # Divide into 2 groups
        labels=['Lower Half', 'Upper Half']
    )

    # Step 2: Extract Distributions of Average Rating for Each Group
    group_distributions = [
        df[df['Ratings Group'] == group]['AverageProfessorRating']
        for group in df['Ratings Group'].unique()
    ]

    return group_distributions

def create_avg_difficulty_group(df):
    median_difficulty = df['Average Difficulty'].median()

    df['Difficulty Groups'] = pd.cut(
    df['Average Difficulty'],
    bins=[df['Average Difficulty'].min(), median_difficulty, df['Average Difficulty'].max()],
    labels=[f'Below Median (≤{median_difficulty:.2f})', f'Above Median (> {median_difficulty:.2f})'],
    include_lowest=True
    )

    groups = df['Difficulty Groups'].unique()
    return groups

def group_avg_difficulty_lavenes_test(groups, df):
    # Step 3: Perform Levene's Test for each pair of groups
    print("Pairwise Levene's Test Results:")

    # Compare groups using nested loops
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]

            # Extract data for the two groups
            group1_data = df[df['Difficulty Groups'] == group1]['AverageProfessorRating']
            group2_data = df[df['Difficulty Groups'] == group2]['AverageProfessorRating']
            
            # Ensure both groups have enough data points
            if len(group1_data) > 1 and len(group2_data) > 1:
                stat, p_value = levene(group1_data, group2_data)
                print(f"Comparing {group1} vs {group2}: Levene's Test Statistic = {stat:.4f}, P-value = {p_value:.4f}")
                
                # Interpretation
                if p_value < 0.005:
                    print("  The variances are significantly different (reject the null hypothesis).")
                else:
                    print("  The variances are not significantly different (fail to reject the null hypothesis).")
            else:
                print(f"Not enough data to compare {group1} vs {group2}.")
            print("-" * 50)


def lavenes_controlled_for_groups(df):

    # Assuming df is loaded with the necessary data

    # Calculate median splits for Average Difficulty and Number of ratings
    median_difficulty = df['Average Difficulty'].median()
    median_ratings = df['NumberOfRatings'].median()

    # Create stratification groups based on the conditions
    df.loc[:, 'Difficulty Groups'] = pd.cut(
        df['Average Difficulty'],
        bins=[df['Average Difficulty'].min(), median_difficulty, df['Average Difficulty'].max()],
        labels=[f'Below Median (≤{median_difficulty:.2f})', f'Above Median (> {median_difficulty:.2f})'],
        include_lowest=True
    )

    df.loc[:, 'Ratings Groups'] = pd.cut(
        df['NumberOfRatings'],
        bins=[df['NumberOfRatings'].min(), median_ratings, df['NumberOfRatings'].max()],
        labels=[f'Below Median (≤{median_ratings:.2f})', f'Above Median (> {median_ratings:.2f})'],
        include_lowest=True
    )

    # Create a combined stratification group
    df.loc[:, 'Stratification Group'] = (
        df['Difficulty Groups'].astype(str) + "_" +
        df['Ratings Groups'].astype(str) + "_" +
        df['Received a pepper'].astype(str)
    )

    # Initialize a list to store Levene's test results and effect sizes
    levene_results = []

    # Get unique stratification groups
    stratification_groups = df['Stratification Group'].unique()

    # Iterate through each stratification group
    for group in stratification_groups:
        # Filter data for males and females in the current group
        male_data = df[(df['HighConfMale'] == 1) & (df['Stratification Group'] == group)]['AverageProfessorRating']
        female_data = df[(df['HighConfFemale'] == 1) & (df['Stratification Group'] == group)]['AverageProfessorRating']
        
        # Calculate sample sizes for the subgroup
        male_sample_size = len(male_data)
        female_sample_size = len(female_data)
        total_sample_size = male_sample_size + female_sample_size
        
        # Ensure both groups have enough data for Levene's test
        if male_sample_size > 1 and female_sample_size > 1:
            stat, p_value = levene(male_data, female_data)
            
            # Calculate Cohen's d for the group
            mean_male = male_data.mean()
            mean_female = female_data.mean()
            std_male = male_data.std()
            std_female = female_data.std()
            pooled_std = np.sqrt(((male_sample_size - 1) * std_male**2 + (female_sample_size - 1) * std_female**2) / (male_sample_size + female_sample_size - 2))
            effect_size = (mean_male - mean_female) / pooled_std
            
            levene_results.append({
                'Stratification Group': group,
                'Levene Stat': stat,
                'P-value': p_value,
                'Significant': p_value < 0.005,  # Using a significance level of 0.005
                'Male Sample Size': male_sample_size,
                'Female Sample Size': female_sample_size,
                'Total Sample Size': total_sample_size,
                'Cohen\'s d': effect_size
            })
        else:
            levene_results.append({
                'Stratification Group': group,
                'Levene Stat': None,
                'P-value': None,
                'Significant': "Insufficient Data",
                'Male Sample Size': male_sample_size,
                'Female Sample Size': female_sample_size,
                'Total Sample Size': total_sample_size,
                'Cohen\'s d': None
            })

    # Convert results to a DataFrame
    levene_results_df = pd.DataFrame(levene_results)

    # Print the results
    print("Levene's Test Results with Cohen's d for Male vs. Female within Subgroups:")
    print(levene_results_df)

        # Bootstrap to calculate the 95% confidence interval for Cohen's d
    bootstrap_effect_sizes = []
    n_bootstrap = 1000

    if not male_data.empty and not female_data.empty:
        for _ in range(n_bootstrap):
            # Resample data with replacement
            male_sample = np.random.choice(male_data, size=len(male_data), replace=True)
            female_sample = np.random.choice(female_data, size=len(female_data), replace=True)

            # Calculate means and standard deviations for resampled data
            mean_male_sample = np.mean(male_sample)
            mean_female_sample = np.mean(female_sample)
            std_male_sample = np.std(male_sample, ddof=1)
            std_female_sample = np.std(female_sample, ddof=1)

            # Calculate pooled standard deviation
            pooled_std_sample = np.sqrt(
                ((len(male_sample) - 1) * std_male_sample**2 + (len(female_sample) - 1) * std_female_sample**2) /
                (len(male_sample) + len(female_sample) - 2)
            )

            # Calculate Cohen's d for the resampled data
            bootstrap_effect_sizes.append((mean_male_sample - mean_female_sample) / pooled_std_sample)

        # Calculate the confidence interval
        lower_bound = np.percentile(bootstrap_effect_sizes, 2.5)
        upper_bound = np.percentile(bootstrap_effect_sizes, 97.5)

        # Print the results
        print(f"95% Confidence Interval for Cohen's d: [{lower_bound:.4f}, {upper_bound:.4f}]")
    else:
        print("No data available for the specified group to perform bootstrap analysis.")


def avg_diff_male_female(df):

    # Function to calculate the effect size (rank-biserial correlation)
    def rank_biserial_effect_size(u_stat, group1, group2):
        n1, n2 = len(group1), len(group2)
        return (2 * u_stat) / (n1 * n2) - 1

    # Function to calculate bootstrap confidence intervals for the effect size
    def bootstrap_effect_size_ci(group1, group2, num_bootstrap=1000, alpha=0.05):
        bootstrapped_effect_sizes = []
        for _ in range(num_bootstrap):
            group1_sample = np.random.choice(group1, size=len(group1), replace=True)
            group2_sample = np.random.choice(group2, size=len(group2), replace=True)
            u_stat_sample, _ = stats.mannwhitneyu(group1_sample, group2_sample, alternative='two-sided')
            effect_size_sample = rank_biserial_effect_size(u_stat_sample, group1_sample, group2_sample)
            bootstrapped_effect_sizes.append(effect_size_sample)
        lower = np.percentile(bootstrapped_effect_sizes, 100 * (alpha / 2))
        upper = np.percentile(bootstrapped_effect_sizes, 100 * (1 - alpha / 2))
        return lower, upper

    # Filter for male and female professor difficulty ratings
    male_difficulty = df[df['HighConfMale'] == 1]['Average Difficulty'].to_numpy()
    female_difficulty = df[df['HighConfFemale'] == 1]['Average Difficulty'].to_numpy()

    # Mann-Whitney U Test
    u_stat, p_value_mw = stats.mannwhitneyu(male_difficulty, female_difficulty, alternative='two-sided')

    # Kolmogorov-Smirnov Test
    ks_stat, p_value_ks = stats.ks_2samp(male_difficulty, female_difficulty)

    # Calculate effect size for Mann-Whitney U
    effect_size_mw = rank_biserial_effect_size(u_stat, male_difficulty, female_difficulty)

    # Bootstrap confidence interval for effect size
    ci_lower, ci_upper = bootstrap_effect_size_ci(male_difficulty, female_difficulty)

    # Display results
    print("Mann-Whitney U Test:")
    print(f"  U-Statistic: {u_stat:.3f}")
    print(f"  P-Value: {p_value_mw:.3f}")
    print(f"  Effect Size (Rank-Biserial Correlation): {effect_size_mw:.3f}")
    print(f"  95% Bootstrap CI for Effect Size: ({ci_lower:.3f}, {ci_upper:.3f})")
    if p_value_mw < 0.005:
        print("  The distributions of average difficulty ratings significantly differ between male and female professors.")
    else:
        print("  The distributions of average difficulty ratings do not significantly differ between male and female professors.")

    print("\nKolmogorov-Smirnov Test:")
    print(f"  KS-Statistic: {ks_stat:.3f}")
    print(f"  P-Value: {p_value_ks:.3f}")
    if p_value_ks < 0.005:
        print("  The distributions of average difficulty ratings significantly differ between male and female professors.")
    else:
        print("  The distributions of average difficulty ratings do not significantly differ between male and female professors.")

def avg_diff_signif_test(df):
    from scipy.stats import ks_2samp

    # Step 1: Calculate the median of 'Number of ratings'
    median_ratings = df['NumberOfRatings'].median()

    # Step 2: Split 'Average Difficulty' into two groups based on the median of 'Number of ratings'
    below_median_difficulty = df[df['NumberOfRatings'] <= median_ratings]['Average Difficulty']
    above_median_difficulty = df[df['NumberOfRatings'] > median_ratings]['Average Difficulty']

    # Step 3: Perform KS test to compare distributions
    ks_stat, p_value = ks_2samp(below_median_difficulty, above_median_difficulty)

    # Display results
    print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Interpretation
    if p_value < 0.005:  # Assuming a significance level of 0.005
        print("The distribution of Average Difficulty significantly changes based on the Number of ratings.")
    else:
        print("The distribution of Average Difficulty does not significantly change based on the Number of ratings.")


def avg_rating_conf(df):
    from scipy.stats import ks_2samp

    # Step 1: Calculate the median of 'Number of ratings'
    median_ratings = df['AverageProfessorRating'].median()

    # Step 2: Split 'Average Difficulty' into two groups based on the median of 'Number of ratings'
    below_median_difficulty = df[df['AverageProfessorRating'] <= median_ratings]['Average Difficulty']
    above_median_difficulty = df[df['AverageProfessorRating'] > median_ratings]['Average Difficulty']

    # Step 3: Perform KS test to compare distributions
    ks_stat, p_value = ks_2samp(below_median_difficulty, above_median_difficulty)

    # Display results
    print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Interpretation
    if p_value < 0.005:  # Assuming a significance level of 0.005
        print("The distribution of Average Difficulty significantly changes based on Average ratings.")
    else:
        print("The distribution of Average Difficulty does not significantly change based on Average ratings.")


def mannwhitney_ks_test(df, column1, column2):

    # Filter for Average Difficulty based on 'Received a pepper'
    pepper_group = df[df[column1] == 1][column2]
    no_pepper_group = df[df[column1] == 0][column2]

    # Mann-Whitney U Test
    u_stat, p_value_mw = stats.mannwhitneyu(pepper_group, no_pepper_group, alternative='two-sided')

    # Kolmogorov-Smirnov Test
    ks_stat, p_value_ks = stats.ks_2samp(pepper_group, no_pepper_group)

    # Display results
    print("Mann-Whitney U Test:")
    print(f"  U-Statistic: {u_stat:.3f}")
    print(f"  P-Value: {p_value_mw:.3f}")

    print("\nKolmogorov-Smirnov Test:")
    print(f"  KS-Statistic: {ks_stat:.3f}")
    print(f"  P-Value: {p_value_ks:.3f}")

def CHI2(df, column1, column2):
    # Create a contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])

    # Perform the chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display the results
    print("Contingency Table:")
    print(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Degrees of Freedom: {dof}")
    print("------------------------------------")
    print("------------------------------------")

def CHI2_MW(df, column1, column2, column3):
    # Iterate over conditions and perform tests
    print("Mann-Whitney U Test and Kolmogorov-Smirnov Test Results:")
    for pepper_status in [0, 1]:  # 0 = No Pepper, 1 = Pepper
        # Filter male and female groups for the current pepper status
        males = df[(df[column1] == 1) &
                    (df[column2] == pepper_status)][column3]
        
        females = df[(df[column1] == 0) &
                        (df[column2] == pepper_status)][column3]
        
        # Check if both groups have enough data
        if len(males) > 1 and len(females) > 1:
            # Mann-Whitney U Test
            u_stat, p_value_mw = stats.mannwhitneyu(males, females, alternative='two-sided')
            
            # Kolmogorov-Smirnov Test
            ks_stat, p_value_ks = stats.ks_2samp(males, females)

            # Print results for this subgroup
            print(f"Group: Pepper = {'Yes' if pepper_status == 1 else 'No'}")
            print(f"  Mann-Whitney U Test Statistic: {u_stat:.3f}")
            print(f"  Mann-Whitney P-Value: {p_value_mw:.3f}")
            print(f"  {'Significant' if p_value_mw < 0.005 else 'Not Significant'}")
            print(f"  Kolmogorov-Smirnov Statistic: {ks_stat:.3f}")
            print(f"  Kolmogorov-Smirnov P-Value: {p_value_ks:.3f}")
            print(f"  {'Significant' if p_value_ks < 0.005 else 'Not Significant'}")
        else:
            # Print message if there is insufficient data
            print(f"Group: Pepper = {'Yes' if pepper_status == 1 else 'No'}")
            print("  Not enough data for Mann-Whitney U Test and Kolmogorov-Smirnov Test")

    # Function to calculate Cohen's d
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Function to calculate bootstrap confidence intervals for Cohen's d
def bootstrap_cohen_d_ci(group1, group2, num_bootstrap=1000, alpha=0.005):
    bootstrapped_d = []
    for _ in range(num_bootstrap):
        # Resample with replacement
        group1_sample = np.random.choice(group1, size=len(group1), replace=True)
        group2_sample = np.random.choice(group2, size=len(group2), replace=True)
        # Calculate Cohen's d for resampled groups
        bootstrapped_d.append(cohen_d(group1_sample, group2_sample))
    # Calculate the confidence intervals
    lower = np.percentile(bootstrapped_d, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_d, 100 * (1 - alpha / 2))
    return lower, upper

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()


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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        #(f"plot_{timestamp}.png")
        #plt.close()
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
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
def scatterdepandindependent(df,dependent_var):
    for var in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[var], y=df[dependent_var])
        plt.title(f'Scatterplot of {dependent_var} vs {var}')
        plt.xlabel(var)
        plt.ylabel(dependent_var)
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        plt.savefig(f"plot_{timestamp}.png")
        plt.close()
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
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
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    
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
    warnings.filterwarnings("ignore", category=FutureWarning)

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
    print("------------------------------------")

    print("Question 2")
    
    print("------------------------------------")
    print("Lavenes test Hypothesis Test: greater than 10 men vs greater than 10 female")
    lavenes_test(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating')
    print("------------------------------------")
    print("95% CI for Lavenes test on Cohens d: greater than 10 men vs greater than 10 female")
    cohen_d_confidence_interval(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, "AverageProfessorRating")
    print("------------------------------------")
    print("Lavenes Test for number of ratings: Split by median")
    grouped_dist = create_num_ratings_group(df_capstone_greater_than_10)
    lavenes_test_group(grouped_dist)
    print("------------------------------------")
    perform_corr_cont_test(df_capstone_greater_than_10, 'Average Difficulty', 'AverageProfessorRating', 'df_capstone_greater_than_10')
    print("------------------------------------")

    print("Lavenes test for Average Difficulty and Average Ratings: Split by median")
    groups_avg_diff = create_avg_difficulty_group(df_capstone_greater_than_10)
    group_avg_difficulty_lavenes_test(groups_avg_diff, df_capstone_greater_than_10)

    print("------------------------------------")

    print("Correlation Avg Difficulty and Average Ratings")

    perform_corr_cont_test(df_capstone_greater_than_10, 'Average Difficulty', 'AverageProfessorRating', 'df_capstone_greater_than_10')

    print("------------------------------------")
    print("Lavenes test for Received a pepper and Average Professor Rating")
    lavenes_test(df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 1], df_capstone_greater_than_10[df_capstone_greater_than_10['Received a pepper'] == 0], 'AverageProfessorRating')
    print("------------------------------------")
    print("Lavenes test after adjusting for confounds")

    lavenes_controlled_for_groups(df_capstone_greater_than_10)

    print("------------------------------------")

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
    
    print("Question 5")
    print("------------------------------------")
    print("Average Difficulty: Male and Females 10 or more ratings")
    avg_diff_male_female(df_capstone_greater_than_10)
    print("------------------------------------")
    print("Average Difficulty and Number of Ratings Correlation")
    # Calculate the Pearson correlation between Average Rating and Average Difficulty
    correlation = df_capstone_greater_than_10[['NumberOfRatings', 'Average Difficulty']].corr().iloc[0, 1]

    # Display the result
    print(f"The correlation between Number of rating and Average Difficulty is: {correlation:.3f}")

    print("------------------------------------")

    avg_diff_signif_test(df_capstone_greater_than_10)
    print("------------------------------------")

    avg_rating_conf(df_capstone_greater_than_10)

    print("------------------------------------")

    # Compute the Pearson correlation between 'Received a Pepper' and 'Average Difficulty'
    correlation = df_capstone_greater_than_10[['Received a pepper', 'Average Difficulty']].corr().iloc[0, 1]

    # Display the result
    print(f"The correlation between 'Received a Pepper' and 'Average Difficulty' is: {correlation:.3f}")
    print("------------------------------------")

    print("Mann Whitney Test for Received a Pepper and Average Difficulty")
    mannwhitney_ks_test(df_capstone_greater_than_10, 'Received a pepper', 'Average Difficulty')
    print("------------------------------------")

    print("Chi Squared Test for Pepper and Gender")    
    CHI2(df_capstone_greater_than_10, 'HighConfMale', 'Received a pepper')
    print("------------------------------------")

    print("Chi Squared and Mann Whitney Test: Controlling for confounds")   
    print("------------------------------------")

    CHI2_MW(df_capstone_greater_than_10, 'HighConfMale', 'Received a pepper', 'Average Difficulty')
    print("------------------------------------")
    print("Question 6")
    print("------------------------------------")
    # Iterate over conditions and calculate Cohen's d with CI
    print("Cohen's d and Bootstrap Confidence Interval Results:")
    for pepper_status in [0, 1]:  # 0 = No Pepper, 1 = Pepper
        # Filter male and female groups for the current pepper status
        males = df_capstone_greater_than_10[(df_capstone_greater_than_10['HighConfMale'] == 1) &
                       (df_capstone_greater_than_10['Received a pepper'] == pepper_status)]['Average Difficulty'].to_numpy()
    
        females = df_capstone_greater_than_10[(df_capstone_greater_than_10['HighConfMale'] == 0) &
                     (df_capstone_greater_than_10['Received a pepper'] == pepper_status)]['Average Difficulty'].to_numpy()
    
        # Check if both groups have enough data
        if len(males) > 1 and len(females) > 1:
            # Calculate Cohen's d
            d = cohen_d(males, females)
        
            # Calculate bootstrap confidence interval for Cohen's d
            ci_lower_d, ci_upper_d = bootstrap_cohen_d_ci(males, females)
        
            # Print results for this subgroup
            print(f"Group: Pepper = {'Yes' if pepper_status == 1 else 'No'}")
            print(f"  Cohen's d (Effect Size): {d:.3f}")
            print(f"  95% Bootstrap CI for Cohen's d: ({ci_lower_d:.3f}, {ci_upper_d:.3f})")
        else:
            # Print message if there is insufficient data
            print(f"Group: Pepper = {'Yes' if pepper_status == 1 else 'No'}")
            print("  Not enough data to calculate Cohen's d and its confidence interval.")
    print("------------------------------------")


    print("Question 7")
    print("Comparing NA vs non NA distributions to help decide if rows should be dropped")
    df_capstone_na_dropped=df_capstone_greater_than_10_all.dropna()
    print(df_capstone_na_dropped['AverageProfessorRating'].mean(),df_capstone_na_dropped['AverageProfessorRating'].median(),df_capstone_na_dropped['AverageProfessorRating'].std())
    df_only_na=df_capstone_greater_than_10_all[df_capstone_greater_than_10_all.isnull().any(axis=1)]
    print(df_only_na['AverageProfessorRating'].mean(),df_only_na['AverageProfessorRating'].median(),df_only_na['AverageProfessorRating'].std())
    sns.boxplot(df_only_na['AverageProfessorRating'])
    plt.title('Distribution of AverageProfessorRating if Proportion column is missing')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    sns.boxplot(df_capstone_na_dropped['AverageProfessorRating'])
    plt.title('Distribution of AverageProfessorRating if Proportion column is present')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(df_capstone_na_dropped['AverageProfessorRating'], bins=30, kde=True, color='blue', label='AverageProfessorRating if prop not missing', stat='density')
    sns.histplot(df_only_na['AverageProfessorRating'], bins=30, kde=True, color='red', label='AverageProfessorRating if prop missing', stat='density')
    plt.title('')
    plt.xlabel('AverageProfessorRating')
    plt.ylabel('Density')
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    df_capstone_na_dropped['Proportion of online class ratings'] = df_capstone_na_dropped['Number of ratings coming from online classes'].div(df_capstone_na_dropped['NumberOfRatings'])
    
    correlation_matrix = df_capstone_na_dropped.corr()

    sns.heatmap(correlation_matrix,cmap = "RdBu_r", annot=True)
    plt.title('Correlation Matrix')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
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
      
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
      
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model', 'Betas','Alpha', 'RMSE', 'R2'])
    results_df=results_df2.drop(columns=['Betas'])
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
      
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    dependent_var='AverageProfessorRating'
    scatterdepandindependent(Q8dfgreater10,dependent_var)
    X = Q8dfgreater10.drop(columns=['AverageProfessorRating','NumberOfRatings'])  # assuming all columns except 'AverageProfessorRating' are features
    y = Q8dfgreater10['AverageProfessorRating']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    mybetas=getbetas(X_train,y_train,X_test,y_test)[0] 
    indices = np.argsort(mybetas)[::-1] 
      
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    X = np.array([
    Q8dfgreater10[0]
    ]).T

    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)
    #print(getbetas(X_train,y_train,X_test,y_test))
    X = np.array([
    Q8dfgreater10[0],
    Q8dfgreater10[1],
    Q8dfgreater10[2],
    Q8dfgreater10[15],
    Q8dfgreater10[16],
    Q8dfgreater10[12],
    ]).T
    y = np.array(Q8dfgreater10['AverageProfessorRating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    
    #print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:])

    print('Question 9')
    
    Q9df=df_capstone[['Average Difficulty','NumberOfRatings']].join(tagsdf, how='inner')

    Q9df.dropna(inplace=True)

    for i in Q9df.columns[2:]:
        Q9df[i] = Q9df[i].div(Q9df['NumberOfRatings'])

    Q9dfgreater10=Q9df[Q9df['NumberOfRatings'] >= 10]


    #print(Q9dfgreater10.corr())
    dependent_var='Average Difficulty'
    scatterdepandindependent(Q9dfgreater10,dependent_var)
    X = Q9dfgreater10.drop(columns=['Average Difficulty','NumberOfRatings'])  # assuming all columns except 'AverageProfessorRating' are features
    y = Q9dfgreater10['Average Difficulty']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetaslasso(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    
    #print(mybetas[indices])
    #print(getbetaslasso(X_train,y_train,X_test,y_test,alpha=1000))
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    mybetas=getbetaslasso(X_train,y_train,X_test,y_test,alpha=1000)[0]
    indices = np.argsort(mybetas)[::-1]
    
    #print(getbetaslasso(X_train,y_train,X_test,y_test,alpha=100))
    plot_betas(getbetaslasso(X_train,y_train,X_test,y_test,alpha=100)[0][1:])

    X = np.array([
        Q9dfgreater10[0],
        Q9dfgreater10[13],
        Q9dfgreater10[6],
        Q9dfgreater10[3],
        Q9dfgreater10[9]
    ]).T

    y = np.array(Q9dfgreater10['Average Difficulty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    
    #print(getbetas(X_train,y_train,X_test,y_test))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results=getKFresults(X_train,y_train)

    results_df2 = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2'])
    results_df = pd.DataFrame(results, columns=['Model','Betas','Alpha', 'RMSE', 'R2']).drop(columns='Betas')
    plot_results(results_df)

    mybetas=getbetas(X_train,y_train,X_test,y_test)[0]
    indices = np.argsort(mybetas)[::-1]
    
    #print(getbetas(X_train,y_train,X_test,y_test))
    plot_betas(getbetas(X_train,y_train,X_test,y_test)[0][1:])

    results=getfinalresults(X_train,y_train,X_test,y_test)
    results_df = pd.DataFrame(results, columns=['Model', 'Alpha', 'RMSE', 'R2'])
    plot_results(results_df)
    

    print("Question 10")

    Q10df=df_capstone_greater_than_10_all.join(tagsdf, how='inner')

    Q10df.dropna(inplace=True)

    for i in Q10df.columns[8:]:
        Q10df[i] = Q10df[i].div(Q10df['NumberOfRatings'])

    Q10df=Q10df[(Q10df['HighConfMale']==1) & (Q10df['HighConfFemale']==0) | (Q10df['HighConfMale']==0) & (Q10df['HighConfFemale']==1)]

    correlation_matrix = Q10df.corr()
    plt.figure(figsize = (40,40))
    sns.heatmap(correlation_matrix,cmap = "RdBu_r", annot=True)
    plt.title('Correlation Matrix')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    averagerating = Q10df['AverageProfessorRating']
    receivedapepper = Q10df['Received a pepper']
    fig, ax = plt.subplots(figsize=(10,6)) # Could also do figure and plt.() later on, but subplots are a generalization

    ax.scatter(x=averagerating, y=receivedapepper, c='purple') # Purple for NYU :)
    ax.set_title("Scatterplot of AverageProfessorRating V Received a pepper")
    ax.set_xlabel("AverageProfessorRating (X)")
    ax.set_ylabel("Received a pepper (Y)")

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    # We definitely shouldn't draw a line through this lol
    # Logistic Regression with one independent variable
    X = Q10df[['AverageProfessorRating']] # Double bracket for shaping (not a worry for multiple logistic regression)
    y = Q10df['Received a pepper']

    # Train-test split from scikit learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )


    # Fit logistic regression
    log_reg_single = LogisticRegression()
    log_reg_single.fit(X_train, y_train)

    # Predictions
    # Class
    y_pred = log_reg_single.predict(X_test)

    # Probabilities
    y_prob = log_reg_single.predict_proba(X_test)[:, 1]

    # Threshold?
    # Get predictions in dataframe
    results = pd.DataFrame({'Predictions': y_pred, 'Probabilities': y_prob})
    print(results.tail())  # Display a few rows

    print(results[results['Predictions'] == 1].min())
    print(results[results['Predictions'] == 0].max())

    THRESHOLD = 0.607 # Revisit later!
    # THRESHOLD = optimal_threshold # from ROC curve
    y_pred_new = (y_prob > THRESHOLD).astype(int)

    # Calculate evaluation metrics
    # Precision = TP / TP + FP
    # Recall = TP / TP + FN
    # F1 = 2 * P * R / (P + R) - harmonic mean, balance precision and recall
    # Support - data pts in the class
    # Macro metrics - average over each class
    # Micro or weighted metrics - treat each sample the same

    class_report = classification_report(y_test, y_pred_new) #y_pred_new
    print(class_report)

    # Interpret coefficients
    print(log_reg_single.coef_)
    # Interpret: For every increase in 1 unit of AverageProfessorRating, we expect the odds of Received a pepper relative to odds of no Received a pepper (the ratio) to increase by e^0.03
    print(np.exp(log_reg_single.coef_[0])) # Slight boost for odds of Received a pepper

    print(log_reg_single.intercept_)
    print(np.exp(log_reg_single.intercept_)) # Not super interpretable, AverageProfessorRating wouldn't be 0. But "base" odds here.
    # If AverageProfessorRating was 0, p / 1 -p or Received a pepper v no Received a pepper would be small odds

    # Confusion Matrix
    conf_matrix_single = confusion_matrix(y_test, y_pred) #y_pred_new
    sns.heatmap(conf_matrix_single, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0 (Did not)", "1 (Received a pepper)"],
                yticklabels=["0 (Did not)", "1 (Received a pepper)"])

    # Add title and labels
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()
    # What do you think doc?

    # Visualize the curve
    # Extract coefficients
    beta1 = log_reg_single.coef_[0][0]
    intercept = log_reg_single.intercept_[0]

    # Apply over training data
    x_vals = np.linspace(X_train.min(), X_train.max(), 100)

    # Logistic reg formula
    y_vals = 1 / (1 + np.exp(-(beta1 * x_vals + intercept)))
    plt.plot(x_vals, y_vals, label="Sigmoid Curve")

    # Add threshold line
    threshold = THRESHOLD
    threshold_x = (np.log(threshold / (1 - threshold)) - intercept) / beta1  # Solve for x when sigmoid(x) = threshold
    plt.axvline(threshold_x, color='red', linestyle='--', label=f'Threshold at AverageProfessorRating = {threshold_x:.2f}')
    plt.title("Visualizing the Curve")
    plt.xlabel("AverageProfessorRating")
    plt.ylabel("Probability of Received a pepper")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    # ROC Curve, get AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    # Try to find optimal threshold for fitting
    optimal_threshold_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]
    print(optimal_threshold)

    # List of chosen thresholds
    chosen_thresholds = [0.607]

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    # Loop over each threshold
    for chosen_threshold in chosen_thresholds:
        # Find index of the threshold closest to the chosen one
        threshold_index = (np.abs(thresholds - chosen_threshold)).argmin()

        # Get the corresponding FPR and TPR for the chosen threshold
        fpr_at_threshold = fpr[threshold_index]
        tpr_at_threshold = tpr[threshold_index]

        # Plot the chosen threshold point on the ROC curve
        plt.scatter(fpr_at_threshold, tpr_at_threshold, label=f"Threshold = {chosen_threshold}", zorder=5)

    # Add title, labels, and legend
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()

    # Show plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    
    X = Q10df.drop(columns=['Received a pepper','NumberOfRatings','Number of ratings coming from online classes','HighConfMale','HighConfFemale',4,8,9,17,18])  # assuming all columns except 'AverageProfessorRating' are features
    X.columns =X.columns.astype(str)
    # Multiple
    # Apply min-max scaling to get variables on same scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X) # Row-wise

    # Logistic Regression with One Explanatory Variable
    y = Q10df['Received a pepper']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )


    # Fit logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predictions
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    # Efficiently create and display the DataFrame
    results = pd.DataFrame({'Predictions': y_pred, 'Probabilities': y_prob})
    print(results.tail())  # Display a few rows

    print(results[results['Predictions'] == 1].min())
    print(results[results['Predictions'] == 0].max())

    THRESHOLD = 0.465 # Revisit later!
    # THRESHOLD = optimal_threshold
    y_pred_new = (y_prob > THRESHOLD).astype(int)

    class_report = classification_report(y_test, y_pred_new) #y_pred_new
    print(class_report)

    print(log_reg.coef_)
    print(np.exp(log_reg.coef_)) # These are huge. But careful, we scaled our X variables (between 0 and 1). Still interpretable over fractional units.

    print(log_reg.intercept_)
    print(np.exp(log_reg.intercept_))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred) #y_pred_new
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0 (No Pepper)", "1 (Pepper)"],
                yticklabels=["0 (No Pepper)", "1 (Pepper)"])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

    optimal_threshold_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]
    print(optimal_threshold)

    # List of chosen thresholds
    chosen_thresholds = [0.488]

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    # Loop over each threshold
    for chosen_threshold in chosen_thresholds:
        # Find index of the threshold closest to the chosen one
        threshold_index = (np.abs(thresholds - chosen_threshold)).argmin()

        # Get the corresponding FPR and TPR for the chosen threshold
        fpr_at_threshold = fpr[threshold_index]
        tpr_at_threshold = tpr[threshold_index]

        # Plot the chosen threshold point on the ROC curve
        plt.scatter(fpr_at_threshold, tpr_at_threshold, label=f"Threshold = {chosen_threshold}", zorder=5)

    # Add title, labels, and legend
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()

    # Show plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    plt.savefig(f"plot_{timestamp}.png")
    plt.close()

            
if __name__ == '__main__':
    main()