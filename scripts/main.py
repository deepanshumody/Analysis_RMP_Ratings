# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene,t, chi2_contingency
import warnings
from math import sqrt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#matplotlib.use('Agg') # Use the 'Agg' backend, which is non-interactive

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

    # Step 1: Calculate the median of 'NumberOfRatings'
    median_ratings = df['AverageProfessorRating'].median()

    # Step 2: Split 'Average Difficulty' into two groups based on the median of 'NumberOfRatings'
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

# -----------------
# Utility Functions
# -----------------

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def normal_regression(X_train, y_train):
    # (X^T X)^(-1) X^T y
    return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

def ridge_regression(X_train, y_train, alpha):
    # (X^T X + alpha * I)^(-1) X^T y
    n_features = X_train.shape[1]
    identity = np.eye(n_features)
    # We often do not regularize the bias term => set identity[0, 0] = 0
    identity[0, 0] = 0
    return np.linalg.inv(X_train.T @ X_train + alpha * identity) @ X_train.T @ y_train

def lasso_regression(X_train, y_train, alpha, max_iter=10000, tol=1e-6):
    """
    Coordinate Descent for Lasso.
    Note: For real-world usage, consider sklearn.linear_model.Lasso.
    """
    m, n = X_train.shape
    beta = np.zeros(n)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(n):
            residual = y_train - X_train @ beta + X_train[:, j] * beta[j]
            rho = X_train[:, j].T @ residual
            if j == 0:  # Intercept (no regularization)
                beta[j] = rho / (X_train[:, j].T @ X_train[:, j])
            else:
                # Soft-thresholding
                beta[j] = np.sign(rho) * max(0, abs(rho) - alpha) / (X_train[:, j].T @ X_train[:, j])
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

def plot_feature_vs_dependent(df, dependent_var):
    """
    Generates scatter plots for each feature in the DataFrame against the dependent variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        dependent_var (str): The dependent variable column name.

    Returns:
        None
    """
    # Exclude the dependent variable from the features
    features = df.columns.drop([dependent_var])

    # Define number of rows and columns for subplots
    n_features = len(features)
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = -(-n_features // n_cols) # Ceiling division
    
    # Set figure size based on rows and columns
    plt.figure(figsize=(20, 5 * n_rows))

    # Create a scatterplot for each feature
    for idx, feature in enumerate(features, start=1):
        plt.subplot(n_rows, n_cols, idx)
        sns.scatterplot(x=df[feature], y=df[dependent_var])
        plt.title(f'{dependent_var} vs {feature}')
        plt.xlabel(feature)
        plt.ylabel(dependent_var)
        plt.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

# Forward Feature Selection

def plot_forward_selection_results(results_df):
    """
    Plots RMSE and R² over the number of features selected.
    """
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
    - max_features: Max number of features to select. If None, selects all.

    Returns:
    - selected_features: List of features selected in order
    - results: List of results for each increment of feature selection
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
            # Subset with current selected + new feature
            X_temp = X[selected_features + [feature]]
            rmse_list = []
            r2_list = []
            
            for train_idx, val_idx in kf.split(X_temp):
                X_train_fold = X_temp.iloc[train_idx]
                X_val_fold   = X_temp.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold   = y.iloc[val_idx]
                
                model = LinearRegression()
                model.fit(X_train_fold, y_train_fold)
                
                y_pred_fold = model.predict(X_val_fold)
                rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                r2_fold   = r2_score(y_val_fold, y_pred_fold)
                
                rmse_list.append(rmse_fold)
                r2_list.append(r2_fold)
            
            avg_rmse = np.mean(rmse_list)
            avg_r2   = np.mean(r2_list)
            
            if avg_rmse < best_rmse:
                best_rmse   = avg_rmse
                best_feature = feature
                best_betas   = model.coef_
                best_alpha   = model.intercept_
        
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        results.append([
            selected_features.copy(),
            best_betas,
            best_alpha,
            best_rmse,
            avg_r2
        ])
        
        if max_features is not None and len(selected_features) >= max_features:
            break

    return selected_features, results

# -----------------
# Main Classes - RegressionAnalysis(Regression) and PepperAnalysis(Classification)
# -----------------

class RegressionAnalysis:
    def __init__(self, X, y, alphas=None, seed=RANDOM_SEED, n_splits=5):
        """
        X, y: features and target arrays
        alphas: array-like of alpha values for Ridge/Lasso
        seed: random seed for reproducibility
        n_splits: number of splits for cross-validation
        """
        self.X = X
        self.y = y
        self.seed = seed
        self.n_splits = n_splits
        self.results_cv = []
        self.results_test = []

        if alphas is None:
            self.alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 
                                    0.1, 1, 2, 5, 10, 20, 100, 1000])
        else:
            self.alphas = alphas

    def cross_validate(self):
        """
        Perform KFold cross-validation on the entire dataset (X, y)
        for Normal, Ridge, Lasso across each alpha (for Ridge/Lasso).
        We store all fold results in self.results_cv as tuples:
          (model_type, alpha, fold_idx, rmse, r2, betas).
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.results_cv = []
        
        fold_num = 1
        for train_index, val_index in kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Add intercept
            X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
            X_val_scaled = np.hstack((np.ones((X_val_scaled.shape[0], 1)), X_val_scaled))

            # 1) Normal Regression
            beta_normal = normal_regression(X_train_scaled, y_train)
            y_pred_normal = X_val_scaled @ beta_normal
            rmse_normal = compute_rmse(y_val, y_pred_normal)
            r2_normal = compute_r2(y_val, y_pred_normal)
            self.results_cv.append(('Normal', None, fold_num, rmse_normal, r2_normal, beta_normal))

            # 2) Ridge for each alpha
            for alpha in self.alphas:
                beta_ridge = ridge_regression(X_train_scaled, y_train, alpha)
                y_pred_ridge = X_val_scaled @ beta_ridge
                rmse_ridge = compute_rmse(y_val, y_pred_ridge)
                r2_ridge = compute_r2(y_val, y_pred_ridge)
                self.results_cv.append(('Ridge', alpha, fold_num, rmse_ridge, r2_ridge, beta_ridge))

            # 3) Lasso for each alpha
            for alpha in self.alphas:
                beta_lasso = lasso_regression(X_train_scaled, y_train, alpha)
                y_pred_lasso = X_val_scaled @ beta_lasso
                rmse_lasso = compute_rmse(y_val, y_pred_lasso)
                r2_lasso = compute_r2(y_val, y_pred_lasso)
                self.results_cv.append(('Lasso', alpha, fold_num, rmse_lasso, r2_lasso, beta_lasso))
            
            fold_num += 1

    def get_cv_results_df(self):
        """Return cross-validation results as a pandas DataFrame."""
        columns = ['Model','Alpha','Fold','RMSE','R2','Betas']
        return pd.DataFrame(self.results_cv, columns=columns)

    def pick_best_alpha(self, model_type='Ridge', metric='RMSE'):
        """
        From cross-validation results, pick the alpha that has the best
        (lowest) mean RMSE or (highest) mean R2 across folds for a given model_type.

        Parameters
        ----------
        model_type : {'Ridge', 'Lasso'}
            The type of model for which to pick best alpha.
        metric : {'RMSE', 'R2'}
            The metric to optimize. 'RMSE' picks the alpha with min RMSE;
            'R2' picks the alpha with max R2.

        Returns
        -------
        best_alpha : float
            The alpha that performed best in cross-validation.
        best_score : float
            The corresponding CV score (RMSE or R2) for that alpha.
        """
        # Pull the cross-validation results into a DataFrame
        df = self.get_cv_results_df()

        # Group by (Model, Alpha), then compute the mean of RMSE and R2 across folds
        df_agg = df.groupby(['Model','Alpha'])[['RMSE','R2']].mean().reset_index()

        # Filter only to the chosen model_type
        # (Make sure user isn't trying to pick alpha for "Normal", which has no alpha)
        if model_type not in ['Ridge', 'Lasso']:
            raise ValueError("model_type must be 'Ridge' or 'Lasso' for picking best alpha.")

        df_agg = df_agg[df_agg['Model'] == model_type]

        if df_agg.empty:
            raise ValueError(f"No cross-validation results found for model '{model_type}'. "
                            f"Check that you ran CV and that your model_type is correct.")

        if metric not in ['RMSE','R2']:
            raise ValueError("metric must be 'RMSE' or 'R2'.")

        # Find the alpha that minimizes or maximizes the chosen metric
        if metric == 'RMSE':
            # We want the alpha with the *lowest* mean RMSE
            best_idx = df_agg['RMSE'].idxmin()
            best_alpha = df_agg.loc[best_idx, 'Alpha']
            best_score = df_agg.loc[best_idx, 'RMSE']
        else:  # metric == 'R2'
            # We want the alpha with the *highest* mean R2
            best_idx = df_agg['R2'].idxmax()
            best_alpha = df_agg.loc[best_idx, 'Alpha']
            best_score = df_agg.loc[best_idx, 'R2']

        return best_alpha, best_score


    def finalize_and_evaluate(self, X_train, y_train, X_test, y_test, 
                              best_ridge_alpha, best_lasso_alpha, 
                              make_residual_plots=True):
        """
        Given the best alpha for Ridge and Lasso (from cross_val),
        train Normal, best Ridge, best Lasso on the entire training set,
        evaluate on the test set, and optionally produce residual plots
        in subplots.
        """
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Add intercept
        X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
        X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

        self.results_test = []

        # 1) Normal
        beta_normal = normal_regression(X_train_scaled, y_train)
        y_pred_normal = X_test_scaled @ beta_normal
        rmse_normal = compute_rmse(y_test, y_pred_normal)
        r2_normal = compute_r2(y_test, y_pred_normal)
        self.results_test.append(('Normal', None, rmse_normal, r2_normal, beta_normal))

        # 2) Ridge (best_ridge_alpha)
        beta_ridge = ridge_regression(X_train_scaled, y_train, best_ridge_alpha)
        y_pred_ridge = X_test_scaled @ beta_ridge
        rmse_ridge = compute_rmse(y_test, y_pred_ridge)
        r2_ridge = compute_r2(y_test, y_pred_ridge)
        self.results_test.append(('Ridge', best_ridge_alpha, rmse_ridge, r2_ridge, beta_ridge))

        # 3) Lasso (best_lasso_alpha)
        beta_lasso = lasso_regression(X_train_scaled, y_train, best_lasso_alpha)
        y_pred_lasso = X_test_scaled @ beta_lasso
        rmse_lasso = compute_rmse(y_test, y_pred_lasso)
        r2_lasso = compute_r2(y_test, y_pred_lasso)
        self.results_test.append(('Lasso', best_lasso_alpha, rmse_lasso, r2_lasso, beta_lasso))

        # Optional: combined residual plots in subplots
        if make_residual_plots:
            # We'll do Normal, Ridge, Lasso side by side
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns

            # Normal
            residuals_normal = y_test - y_pred_normal
            std_res_normal = (residuals_normal - residuals_normal.mean()) / residuals_normal.std()
            axs[0, 0].scatter(y_pred_normal, std_res_normal, color='blue')
            axs[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axs[0, 0].set_title("Normal Residual Plot")
            axs[0, 0].set_xlabel("Predicted")
            axs[0, 0].set_ylabel("Std Residuals")
            axs[1, 0].hist(residuals_normal, bins=15, color='green', edgecolor='black', density=True)
            axs[1, 0].set_title("Normal Residual Histogram")

            # Ridge
            residuals_ridge = y_test - y_pred_ridge
            std_res_ridge = (residuals_ridge - residuals_ridge.mean()) / residuals_ridge.std()
            axs[0, 1].scatter(y_pred_ridge, std_res_ridge, color='blue')
            axs[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axs[0, 1].set_title(f"Ridge (alpha={best_ridge_alpha}) Residual")
            axs[0, 1].set_xlabel("Predicted")
            axs[0, 1].set_ylabel("Std Residuals")
            axs[1, 1].hist(residuals_ridge, bins=15, color='green', edgecolor='black', density=True)
            axs[1, 1].set_title("Ridge Residual Histogram")

            # Lasso
            residuals_lasso = y_test - y_pred_lasso
            std_res_lasso = (residuals_lasso - residuals_lasso.mean()) / residuals_lasso.std()
            axs[0, 2].scatter(y_pred_lasso, std_res_lasso, color='blue')
            axs[0, 2].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axs[0, 2].set_title(f"Lasso (alpha={best_lasso_alpha}) Residual")
            axs[0, 2].set_xlabel("Predicted")
            axs[0, 2].set_ylabel("Std Residuals")
            axs[1, 2].hist(residuals_lasso, bins=15, color='green', edgecolor='black', density=True)
            axs[1, 2].set_title("Lasso Residual Histogram")

            plt.tight_layout()
            plt.show()

    def get_test_results_df(self):
        """Return final test results as a DataFrame."""
        columns = ['Model','Alpha','RMSE','R2','Betas']
        return pd.DataFrame(self.results_test, columns=columns)

    def plot_cv_rmse_r2(self):
        """
        Plot average CV RMSE and R^2 vs. alpha on log scale.
        """
        df = self.get_cv_results_df()
        df_agg = df.groupby(['Model','Alpha'])[['RMSE','R2']].mean().reset_index()

        normal_subset = df_agg[df_agg['Model'] == 'Normal']
        ridge_subset = df_agg[df_agg['Model'] == 'Ridge']
        lasso_subset = df_agg[df_agg['Model'] == 'Lasso']

        # RMSE plot
        plt.figure(figsize=(10,6))
        plt.xscale('log')

        plt.plot(ridge_subset['Alpha'], ridge_subset['RMSE'], marker='o', label='Ridge RMSE')
        plt.plot(lasso_subset['Alpha'], lasso_subset['RMSE'], marker='s', label='Lasso RMSE')
        if len(normal_subset) > 0:
            normal_rmse = normal_subset['RMSE'].mean()
            plt.axhline(y=normal_rmse, color='r', linestyle='--', label=f'Normal RMSE={normal_rmse:.3f}')

        plt.title('CV RMSE vs. Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.show()

        # R2 plot
        plt.figure(figsize=(10,6))
        plt.xscale('log')

        plt.plot(ridge_subset['Alpha'], ridge_subset['R2'], marker='o', label='Ridge R2')
        plt.plot(lasso_subset['Alpha'], lasso_subset['R2'], marker='s', label='Lasso R2')
        if len(normal_subset) > 0:
            normal_r2 = normal_subset['R2'].mean()
            plt.axhline(y=normal_r2, color='r', linestyle='--', label=f'Normal R2={normal_r2:.3f}')

        plt.title('CV R2 vs. Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('R2')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_coefs(self, betas, feature_names=None, model_name='Normal', alpha=None):

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(betas))]
        
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, betas)
        
        if model_name == 'Normal':
            plt.title("Coefficients: Normal Regression")
        else:
            if alpha is not None:
                plt.title(f"Coefficients: {model_name} (alpha={alpha})")
            else:
                plt.title(f"Coefficients: {model_name}")

        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class PepperAnalysis:
    def __init__(
        self, 
        df_capstone, 
        tagsdf, 
        seed=RANDOM_SEED
    ):
        """
        Initialize PepperAnalysis with the two dataframes and a random seed.
        df_capstone: main DataFrame with course/professor data
        tagsdf: additional DataFrame to be joined
        """
        self.seed = seed
        self.df_capstone = df_capstone
        self.tagsdf = tagsdf
        self.tagsdf.columns=self.tagsdf.columns.astype(str)
        self.df = None  # Will hold the merged/cleaned DataFrame later
        
        # Placeholders for trained models (if you want to access them outside)
        self.log_reg_single = None
        self.log_reg_multi = None
        self.svm_model = None

    def preprocess_data(self):
        """
        Merge (inner join) df_capstone and tagsdf, drop NaN, compute proportions,
        filter for Male or Female professor, etc. 
        Stores the cleaned data in self.df.
        """
        # 1) Merge
        Q10df = self.df_capstone.join(self.tagsdf, how='inner')

        # 2) Drop missing
        Q10df.dropna(inplace=True)

        # 3) Convert columns i >= 8 to proportion by dividing by "NumberOfRatings"
        for i in Q10df.columns[8:]:
            Q10df[i] = Q10df[i].div(Q10df['NumberOfRatings'])

        # 4) Filter rows to (Male=1,Female=0) or (Male=0,Female=1)
        #    means "exactly one of them is 1"
        #    The parentheses for & | are crucial.
        Q10df = Q10df[((Q10df['HighConfMale'] == 1) & (Q10df['HighConfFemale'] == 0)) |
                      ((Q10df['HighConfMale'] == 0) & (Q10df['HighConfFemale'] == 1))]

        self.df = Q10df.copy()  # store cleaned DataFrame in self.df

    def plot_correlation_matrix(self):
        """
        Plots a large correlation matrix for self.df, if you want to see all features.
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")
        
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(40, 40))
        sns.heatmap(correlation_matrix, cmap="RdBu_r", annot=True)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_scatter_single(self, x_col='Average Rating', y_col='Received a pepper'):
        """
        Simple scatter plot of x_col vs. y_col in self.df.
        By default: x=Average Rating, y=Received a pepper
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")

        x_vals = self.df[x_col]
        y_vals = self.df[y_col]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x=x_vals, y=y_vals, c='purple')
        ax.set_title(f"Scatterplot of {x_col} vs. {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        plt.show()

    def logistic_regression_single_var(self, x_col='Average Rating', y_col='Received a pepper', threshold=0.5):
        """
        Fits a single-variable Logistic Regression using x_col as predictor for y_col.
        Plots confusion matrix, classification report, ROC, etc.
        
        threshold: The cutoff for assigning class = 1 if P(Y=1) > threshold.
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")

        X = self.df[[x_col]]  # must be 2D => double bracket
        y = self.df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # Fit logistic regression
        log_reg_single = LogisticRegression()
        log_reg_single.fit(X_train, y_train)
        self.log_reg_single = log_reg_single  # store model

        # Predictions
        y_pred = log_reg_single.predict(X_test)
        y_prob = log_reg_single.predict_proba(X_test)[:, 1]

        # Thresholding
        y_pred_new = (y_prob > threshold).astype(int)

        # Classification report
        class_report = classification_report(y_test, y_pred_new)
        print("Classification Report (Single-Var Logistic):")
        print(class_report)

        # Coefficients
        beta1 = log_reg_single.coef_[0][0]
        intercept = log_reg_single.intercept_[0]
        print(f"Coefficient (beta1): {beta1:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"Odds multiplier (exp(beta1)): {np.exp(beta1):.4f}")
        print(f"Intercept as odds => exp(intercept): {np.exp(intercept):.4f}")

        # Confusion Matrix
        conf_matrix_single = confusion_matrix(y_test, y_pred_new)
        sns.heatmap(
            conf_matrix_single, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0 (No Pepper)", "1 (Pepper)"],
            yticklabels=["0 (No Pepper)", "1 (Pepper)"]
        )
        plt.title("Confusion Matrix (Single-Var Logistic)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Plot the Sigmoid curve
        x_array = np.linspace(X_train.min()[0], X_train.max()[0], 100)
        # logistic function
        sig = 1 / (1 + np.exp(-(beta1 * x_array + intercept)))
        plt.plot(x_array, sig, label="Sigmoid Curve")

        # Mark the threshold point
        # Solve for x when sigmoid = threshold => x = [ln(threshold/(1-threshold)) - intercept]/beta1
        threshold_x = (np.log(threshold / (1 - threshold)) - intercept) / beta1
        plt.axvline(threshold_x, color='red', linestyle='--',
                    label=f'Threshold at x={threshold_x:.2f}')

        plt.title("Single-Var Logistic Sigmoid Curve")
        plt.xlabel(x_col)
        plt.ylabel("Probability of Pepper (y=1)")
        plt.legend()
        plt.show()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve (Single-Var Logistic)")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.legend()
        plt.show()

        # If you want the "optimal threshold" from Youden's J statistic (tpr - fpr)
        optimal_threshold_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_index]
        print(f"Optimal Threshold (Youden's J): {optimal_threshold:.3f}")

    def logistic_regression_multi_var(self, drop_cols=None, threshold=0.5):
        """
        Fits a multi-variable logistic regression on self.df.
        You can specify columns to drop (like 'Received a pepper', 'NumberOfRatings', etc.).
        We apply MinMax scaling and do a train/test split, then show classification metrics, confusion matrix, ROC, etc.
        
        threshold: Probability threshold for predicting class=1
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")

        if drop_cols is None:
            # By default, let's drop these columns to avoid target leakage or data not needed
            drop_cols = ['Received a pepper', 'NumberOfRatings', 
                         'Number of ratings coming from online classes',
                         'HighConfMale', 'HighConfFemale']

        # X => all columns except what's in drop_cols
        # but also be mindful of columns that might be numeric only
        df_clean = self.df.drop(columns=drop_cols, errors='ignore')

        # The target
        y = self.df['Received a pepper']

        # Convert X to numeric
        X = df_clean.select_dtypes(include=[np.number])

        # Scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.seed
        )

        # Fit logistic regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        self.log_reg_multi = log_reg  # store model

        # Predictions
        y_pred = log_reg.predict(X_test)
        y_prob = log_reg.predict_proba(X_test)[:, 1]

        # Thresholding
        y_pred_new = (y_prob > threshold).astype(int)

        # Classification report
        class_report = classification_report(y_test, y_pred_new)
        print("\nClassification Report (Multi-Var Logistic):")
        print(class_report)

        print("\nCoefficients (log scale):")
        print(log_reg.coef_)
        print("Exp of Coefficients (odds multipliers):")
        print(np.exp(log_reg.coef_))
        print("Intercept:", log_reg.intercept_)
        print("Intercept (exp):", np.exp(log_reg.intercept_))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_new)
        sns.heatmap(
            conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0 (No Pepper)", "1 (Pepper)"],
            yticklabels=["0 (No Pepper)", "1 (Pepper)"]
        )
        plt.title("Confusion Matrix (Multi-Var Logistic)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve (Multi-Var Logistic)")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.legend()
        plt.show()

        # If you want the "optimal threshold" from Youden's J statistic
        optimal_threshold_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_index]
        print(f"Optimal Threshold (Youden's J) for multi-var logistic: {optimal_threshold:.3f}")

    def train_svm(self, drop_cols=None):
        """
        Train a linear SVM on self.df for classification (pepper vs. no pepper),
        then print classification report and plot ROC curve.
        """
        if self.df is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")

        if drop_cols is None:
            drop_cols = ['Received a pepper', 'NumberOfRatings', 
                         'Number of ratings coming from online classes',
                         'HighConfMale', 'HighConfFemale']

        y = self.df['Received a pepper']
        df_clean = self.df.drop(columns=drop_cols, errors='ignore')
        X = df_clean.select_dtypes(include=[np.number])

        # Scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.seed
        )

        svm_model = SVC(kernel='linear', probability=True, random_state=self.seed)
        svm_model.fit(X_train, y_train)
        self.svm_model = svm_model  # store model

        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:, 1]

        print("Classification Report (SVM):")
        print(classification_report(y_test, y_pred))

        # Coefficients
        w = svm_model.coef_[0]
        b = svm_model.intercept_[0]
        print("\nSVM Coefficients (w):", w)
        print("SVM Intercept (b):", b)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve (SVM)")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.legend()
        plt.show()
        
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Question 1")
    df_capstone = pd.read_csv('./rmpCapstoneNum.csv', header=0)
    tagsdf=pd.read_csv('./rmpCapstoneTags.csv', header=0)
    df_capstone.columns = ['AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper', 
                       'Proportion of students that said they would take the class again', 
                       'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale']
    tagsdf.columns = list(range(20))
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
    df_merged_min_10.iloc[:, 5:] = df_merged_min_10.iloc[:, 5:].div(df_merged_min_10['NumberOfRatings'], axis=0)

    p_vals_min_10 = create_p_vals_df(df_merged_min_10)
    visualize_p_vals(p_vals_min_10, 'For professors with more than 10 ratings')
    print_pvals(p_vals_min_10)
    print("------------------------------------")
    print("For professors with more than 19 ratings and no pepper rating")
    df_merged_min_19_no_pepper = df_merged_min_10[(df_merged_min_10['Received a pepper'] == 0) & (df_merged_min_10['NumberOfRatings'] >= 19)]
    p_vals_min_19_no_pepper = create_p_vals_df(df_merged_min_19_no_pepper)
    visualize_p_vals(p_vals_min_19_no_pepper, 'For professors with more than 19 ratings and no pepper rating')
    print_pvals(p_vals_min_19_no_pepper)
    #visualize_density_plot(df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfMale'] == 1], df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfFemale'] == 1], 'Pop quizzes!', 'HighConfMale', 'HighConfFemale', nbins = 10)
    #print("Median: df_merged_min_19_no_pepper male", df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfMale'] == 1]['Pop quizzes!'].describe(), "Median: df_merged_min_19_no_pepper female", df_merged_min_19_no_pepper[df_merged_min_19_no_pepper['HighConfFemale'] == 1]['Pop quizzes!'].describe())
    print("------------------------------------")
    
    print("Question 5")
    print("------------------------------------")
    print("Average Difficulty: Male and Females 10 or more ratings")
    avg_diff_male_female(df_capstone_greater_than_10)
    print("------------------------------------")
    print("Average Difficulty and Number of Ratings Correlation")
    # Calculate the Pearson correlation between Average Rating and Average Difficulty
    correlation = df_capstone_greater_than_10[['NumberOfRatings', 'Average Difficulty']].corr().iloc[0, 1]

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
    print("------------------------------------")

    df_capstone_greater_than_10=df_capstone_greater_than_10.drop(columns=['Ratings Group', 'Difficulty Groups',
       'Ratings Groups', 'Stratification Group'])
    print(df_capstone_greater_than_10.describe())
    print("Missing values per column:\n", df_capstone_greater_than_10_all.isna().sum())

    # Split data into rows with missing values and rows without missing values
    df_no_na = df_capstone_greater_than_10.dropna().copy()
    df_missing = df_capstone_greater_than_10[df_capstone_greater_than_10.isnull().any(axis=1)].copy()

    # -----------------------------
    # 3. Descriptive Statistics
    # -----------------------------

    print("\n--- Descriptive Statistics (No Missing) ---")
    print(f"Mean:   {df_no_na['AverageProfessorRating'].mean():.3f}")
    print(f"Median: {df_no_na['AverageProfessorRating'].median():.3f}")
    print(f"Std:    {df_no_na['AverageProfessorRating'].std():.3f}")

    print("\n--- Descriptive Statistics (Missing) ---")
    print(f"Mean:   {df_missing['AverageProfessorRating'].mean():.3f}")
    print(f"Median: {df_missing['AverageProfessorRating'].median():.3f}")
    print(f"Std:    {df_missing['AverageProfessorRating'].std():.3f}")

    # -----------------------------
    # 4. Boxplots Comparing Groups
    # -----------------------------

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_missing['AverageProfessorRating'], color='red')
    plt.title('AverageProfessorRating if Proportion column is missing')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_no_na['AverageProfessorRating'], color='blue')
    plt.title('AverageProfessorRating if Proportion column is present')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 5. Histogram Comparison
    # -----------------------------

    plt.figure(figsize=(8, 5))
    sns.histplot(df_no_na['AverageProfessorRating'], 
                bins=30, kde=True, color='blue', 
                label='Proportion not missing', 
                stat='density')

    sns.histplot(df_missing['AverageProfessorRating'], 
                bins=30, kde=True, color='red', 
                label='Proportion missing', 
                stat='density')

    plt.xlabel('Average Professor Rating')
    plt.ylabel('Density')
    plt.legend()
    plt.title('AverageProfessorRatings by Missing vs. Non-Missing Proportion')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 6. Two-Sample Tests
    # -----------------------------

    ks_stat, ks_p_val = stats.ks_2samp(df_no_na['AverageProfessorRating'], df_missing['AverageProfessorRating'])
    mw_stat, mw_p_val = stats.mannwhitneyu(df_no_na['AverageProfessorRating'], df_missing['AverageProfessorRating'])

    print("\n--- Two-Sample Tests (AverageProfessorRating) ---")
    print(f"KS Test:           Statistic={ks_stat:.3f}, P-value={ks_p_val:.3f}")
    print(f"Mann-Whitney Test: Statistic={mw_stat:.3f}, P-value={mw_p_val:.3f}")

    # -----------------------------
    # 7. Summary Stats of No-NA Data
    # -----------------------------

    print("\n--- Summary Stats: df_no_na ---")
    print(df_no_na.describe())

    # -----------------------------
    # 8. Add Proportion of Online Class Ratings
    # -----------------------------

    df_no_na['Proportion of online class ratings'] = (
        df_no_na['Number of ratings coming from online classes'] /
        df_no_na['NumberOfRatings']
    )

    # -----------------------------
    # 9. Correlation Matrix
    # -----------------------------
    correlation_matrix = df_no_na.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, cmap="RdBu_r", annot=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 10. Final DataFrame Filtering
    # -----------------------------

    # Keep rows where only male OR female professor
    df_capstone_dropped_final = df_no_na[
        ((df_no_na['HighConfMale'] == 1) & (df_no_na['HighConfFemale'] == 0)) |
        ((df_no_na['HighConfMale'] == 0) & (df_no_na['HighConfFemale'] == 1))
    ]

    # Display the final DataFrame
    print("\n--- Final Filtered DataFrame ---")
    print(df_capstone_dropped_final.head())
    print(f"Final DF Shape: {df_capstone_dropped_final.shape}")

    dependent_var='AverageProfessorRating'

    plot_feature_vs_dependent(df_capstone_dropped_final, dependent_var)
    
    feature_subsets = [
    [
        'Proportion of students that said they would take the class again'
    ],
    [
        'Average Difficulty',
        'Received a pepper',
        'Proportion of students that said they would take the class again'
    ],
    [
        'Average Difficulty',
        'NumberOfRatings',
        'Received a pepper',
        'HighConfFemale',
        'Proportion of online class ratings',
        'Proportion of students that said they would take the class again'
    ]]



    # Your target column
    target_col = 'AverageProfessorRating'

    # Split into train/test
    df_train, df_test = train_test_split(df_capstone_dropped_final, 
                                        test_size=0.2, 
                                        random_state=RANDOM_SEED)

    # We'll store final results for each subset in a list or dictionary
    all_subsets_results = []

    for idx, subset_cols in enumerate(feature_subsets, start=1):
        print(f"--- Subset #{idx}: {subset_cols} ---")
        
        # 1) Create X_train, y_train, X_test, y_test
        X_train = df_train[subset_cols].to_numpy()
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[subset_cols].to_numpy()
        y_test = df_test[target_col].to_numpy()
        
        # 2) Instantiate RegressionAnalysis for THIS subset
        alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 
                        0.1, 1, 2, 5, 10, 20, 100, 1000, 2000, 10000])
        
        analysis = RegressionAnalysis(
            X_train, y_train,
            alphas=alphas,
            seed=RANDOM_SEED,
            n_splits=5  # K-folds
        )
        
        
        # 3) Cross-validate on the TRAIN set
        analysis.cross_validate()
        
        analysis.plot_cv_rmse_r2()

        # 4) Pick best alpha for Ridge and Lasso based on RMSE from CV
        best_ridge_alpha, best_ridge_rmse = analysis.pick_best_alpha('Ridge', metric='RMSE')
        best_lasso_alpha, best_lasso_rmse = analysis.pick_best_alpha('Lasso', metric='RMSE')
        
        print(f"Best Ridge alpha (subset #{idx}): {best_ridge_alpha} with mean CV-RMSE={best_ridge_rmse:.3f}")
        print(f"Best Lasso alpha (subset #{idx}): {best_lasso_alpha} with mean CV-RMSE={best_lasso_rmse:.3f}")
        
        # 5) Train final models on the full TRAIN set, evaluate once on TEST
        #    This will store results in analysis.results_test
        analysis.finalize_and_evaluate(
            X_train, y_train,
            X_test,  y_test,
            best_ridge_alpha,
            best_lasso_alpha,
            make_residual_plots=False  # Set to True if you want residual subplots
        )
        
        final_test_df = analysis.get_test_results_df()
        print("Final Test Results:\n", final_test_df)
        
        # 6) Optionally, plot coefficients for each final model
        #    a) Normal
        normal_row = final_test_df[final_test_df['Model'] == 'Normal'].iloc[0]
        betas_normal = normal_row['Betas']
        analysis.plot_coefs(
            betas_normal[1:], 
            feature_names=subset_cols,
            model_name=f'Normal (Subset #{idx})'
        )
        
        #    b) Best Ridge
        ridge_row = final_test_df[
            (final_test_df['Model'] == 'Ridge') & 
            (final_test_df['Alpha'] == best_ridge_alpha)
        ].iloc[0]
        betas_ridge = ridge_row['Betas']
        analysis.plot_coefs(
            betas_ridge[1:], 
            feature_names=subset_cols,
            model_name=f'Ridge (Subset #{idx})', 
            alpha=best_ridge_alpha
        )
        
        #    c) Best Lasso
        lasso_row = final_test_df[
            (final_test_df['Model'] == 'Lasso') & 
            (final_test_df['Alpha'] == best_lasso_alpha)
        ].iloc[0]
        betas_lasso = lasso_row['Betas']
        analysis.plot_coefs(
            betas_lasso[1:], 
            feature_names=subset_cols,
            model_name=f'Lasso (Subset #{idx})', 
            alpha=best_lasso_alpha
        )
        
        # Save or store this subset's final results for later
        all_subsets_results.append({
            'subset_index': idx,
            'subset_columns': subset_cols,
            'cv_results': analysis.get_cv_results_df(),  # Cross-validation data
            'test_results': final_test_df                # Final test results
        })
        
        print("------------------------------------------------------\n")
        
        print("All subsets processed and evaluated!")
        
    print("Question 8")
    print("------------------------------------")

    # Join tags with df_capstone based on shared index
    Q8df = df_capstone_greater_than_10_all[['AverageProfessorRating', 'NumberOfRatings']].join(tagsdf)
    
    # Drop rows with missing values
    Q8df.dropna(inplace=True)

    # Convert tag counts to proportions by dividing each tag count by total ratings
    for col in Q8df.columns[2:]:
        Q8df[col] = Q8df[col].div(Q8df['NumberOfRatings'])

    # Subset to rows with at least 10 ratings
    Q8dfgreater10 = Q8df[Q8df['NumberOfRatings'] >= 10]
    # Identify pairs of columns with correlation > 0.42
    corr_matrix = Q8dfgreater10.corr()
    corrgreater40 = corr_matrix[corr_matrix > 0.42]
    high_corr_pairs = [
        pair for pair in list(corrgreater40[corrgreater40.notnull()].stack().index)
        if pair[0] != pair[1]
    ]
    print("Highly correlated pairs (corr > 0.42):", high_corr_pairs)

    # Heatmap of correlations (excluding 'NumberOfRatings')
    correlation_matrix = Q8dfgreater10.drop(columns=['NumberOfRatings']).corr()
    plt.figure(figsize=(40, 40))
    sns.heatmap(correlation_matrix, cmap="RdBu_r", annot=True)
    plt.title('Correlation Matrix (Tags Only)')
    plt.show()

    # Scatter plots of each feature vs. AverageProfessorRating in a single plot with subplots
    dependent_var = 'AverageProfessorRating'

    plot_feature_vs_dependent(Q8dfgreater10, dependent_var)

    # Prepare data for feature selection
    X = Q8dfgreater10.drop(columns=['AverageProfessorRating', 'NumberOfRatings'])
    y = Q8dfgreater10['AverageProfessorRating']
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    # Forward feature selection (K-fold CV)
    selected_features_kfold, forward_results_kfold = forward_feature_selection_kfold(X_train, y_train, k=5)
    # Convert results to DataFrame
    forward_results_kfold_df = pd.DataFrame(
        forward_results_kfold,
        columns=['Selected Features', 'Betas', 'Alpha', 'RMSE', 'R2']
    )
    print("\n--- Forward Feature Selection Results (K-fold CV) ---")
    print(forward_results_kfold_df)
    print(tagsdf)
    print(df_capstone_greater_than_10_all)

    # Optionally remove Betas for cleaner display
    forward_results_kfold_df_cleaned = forward_results_kfold_df.drop(columns='Betas')

    # Plot RMSE and R² vs number of features
    plot_forward_selection_results(forward_results_kfold_df_cleaned)

    # Evaluate the final model on the test set
    best_selected_features_kfold = forward_results_kfold_df.iloc[-1]['Selected Features']
    X_train_best_kfold = X_train[best_selected_features_kfold]
    X_test_best_kfold  = X_test[best_selected_features_kfold]

    final_model_kfold = LinearRegression()
    final_model_kfold.fit(X_train_best_kfold, y_train)
    y_pred_test_kfold = final_model_kfold.predict(X_test_best_kfold)

    final_rmse_kfold = np.sqrt(mean_squared_error(y_test, y_pred_test_kfold))
    final_r2_kfold   = r2_score(y_test, y_pred_test_kfold)

    print("\n----- Final Model Evaluation on Test Set -----")
    print(f"Selected Features: {best_selected_features_kfold}")
    print(f"Final Model RMSE: {final_rmse_kfold:.4f}")
    print(f"Final Model R²:   {final_r2_kfold:.4f}")

    forward_results_kfold_df
        
    feature_subsets = [
    list(range(20)), #0 to 19 features
    [2],#TODO Try sqrt, sq, cube etc,
    [0, 2, 1, 15, 16, 12]
    ]


    # Your target column
    target_col = 'AverageProfessorRating'

    # We'll do a single train/test split once at the beginning
    df_train, df_test = train_test_split(Q8dfgreater10, 
                                        test_size=0.2, 
                                        random_state=RANDOM_SEED)

    # We'll store final results for each subset in a list or dictionary
    all_subsets_results = []

    for idx, subset_cols in enumerate(feature_subsets, start=1):
        print(f"--- Subset #{idx}: {subset_cols} ---")
        
        # 1) Create X_train, y_train, X_test, y_test
        X_train = df_train[subset_cols].to_numpy()
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[subset_cols].to_numpy()
        y_test = df_test[target_col].to_numpy()
        
        # 2) Instantiate RegressionAnalysis for THIS subset
        alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 
                        0.1, 1, 2, 5, 10, 20, 100, 1000, 2000, 10000])
        
        analysis = RegressionAnalysis(
            X_train, y_train,
            alphas=alphas,
            seed=RANDOM_SEED,
            n_splits=5  # K-folds
        )
        
        
        # 3) Cross-validate on the TRAIN set
        analysis.cross_validate()
        
        analysis.plot_cv_rmse_r2()

        # 4) Pick best alpha for Ridge and Lasso based on RMSE from CV
        best_ridge_alpha, best_ridge_rmse = analysis.pick_best_alpha('Ridge', metric='RMSE')
        best_lasso_alpha, best_lasso_rmse = analysis.pick_best_alpha('Lasso', metric='RMSE')
        
        print(f"Best Ridge alpha (subset #{idx}): {best_ridge_alpha} with mean CV-RMSE={best_ridge_rmse:.3f}")
        print(f"Best Lasso alpha (subset #{idx}): {best_lasso_alpha} with mean CV-RMSE={best_lasso_rmse:.3f}")
        
        # 5) Train final models on the full TRAIN set, evaluate once on TEST
        #    This will store results in analysis.results_test
        analysis.finalize_and_evaluate(
            X_train, y_train,
            X_test,  y_test,
            best_ridge_alpha,
            best_lasso_alpha,
            make_residual_plots=False  # Set to True if you want residual subplots
        )
        
        final_test_df = analysis.get_test_results_df()
        print("Final Test Results:\n", final_test_df)
        
        # 6) Optionally, plot coefficients for each final model
        #    a) Normal
        normal_row = final_test_df[final_test_df['Model'] == 'Normal'].iloc[0]
        betas_normal = normal_row['Betas']
        analysis.plot_coefs(
            betas_normal[1:], 
            feature_names=subset_cols,
            model_name=f'Normal (Subset #{idx})'
        )
        
        #    b) Best Ridge
        ridge_row = final_test_df[
            (final_test_df['Model'] == 'Ridge') & 
            (final_test_df['Alpha'] == best_ridge_alpha)
        ].iloc[0]
        betas_ridge = ridge_row['Betas']
        analysis.plot_coefs(
            betas_ridge[1:], 
            feature_names=subset_cols,
            model_name=f'Ridge (Subset #{idx})', 
            alpha=best_ridge_alpha
        )
        
        #    c) Best Lasso
        lasso_row = final_test_df[
            (final_test_df['Model'] == 'Lasso') & 
            (final_test_df['Alpha'] == best_lasso_alpha)
        ].iloc[0]
        betas_lasso = lasso_row['Betas']
        analysis.plot_coefs(
            betas_lasso[1:], 
            feature_names=subset_cols,
            model_name=f'Lasso (Subset #{idx})', 
            alpha=best_lasso_alpha
        )
        
        # Save or store this subset's final results for later
        all_subsets_results.append({
            'subset_index': idx,
            'subset_columns': subset_cols,
            'cv_results': analysis.get_cv_results_df(),  # Cross-validation data
            'test_results': final_test_df                # Final test results
        })
        
        print("------------------------------------------------------\n")

    print("All subsets processed and evaluated!")
    print("Question 9")
    print("------------------------------------")
    # Prepare the data
    Q9df = df_capstone_greater_than_10_all[['Average Difficulty', 'NumberOfRatings']].join(tagsdf, how='inner')
    Q9df.dropna(inplace=True)

    # Normalize tags by 'NumberOfRatings'
    for column in Q9df.columns[2:]:
        Q9df[column] = Q9df[column].div(Q9df['NumberOfRatings'])

    # Filter rows with 'NumberOfRatings' >= 10
    Q9dfgreater10 = Q9df[Q9df['NumberOfRatings'] >= 10]

    # Correlation analysis
    correlation_matrix = Q9dfgreater10.drop(columns=['NumberOfRatings']).corr()
    plt.figure(figsize=(40, 40))
    sns.heatmap(correlation_matrix, cmap="RdBu_r", annot=True)
    plt.title('Correlation Matrix (Tags Only)')
    plt.show()

    # Generate scatter plots of features vs. 'Average Difficulty'
    dependent_var = 'Average Difficulty'
    plot_feature_vs_dependent(Q9dfgreater10, dependent_var)

    # Prepare data for modeling
    X = Q9dfgreater10.drop(columns=['Average Difficulty', 'NumberOfRatings'])
    y = Q9dfgreater10['Average Difficulty']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Forward feature selection with K-fold cross-validation
    selected_features_kfold, forward_results_kfold = forward_feature_selection_kfold(X_train, y_train, k=5)

    # Convert results to DataFrame for analysis
    forward_results_kfold_df = pd.DataFrame(
        forward_results_kfold,
        columns=['Selected Features', 'Betas', 'Alpha', 'RMSE', 'R2']
    )

    # Optional: Clean up the results by removing the 'Betas' column
    forward_results_kfold_df_cleaned = forward_results_kfold_df.drop(columns='Betas')

    # Plot RMSE and R² vs number of features
    plot_forward_selection_results(forward_results_kfold_df_cleaned)

    # Evaluate the final model on the test set
    best_selected_features_kfold = forward_results_kfold_df.iloc[-1]['Selected Features']
    X_train_best_kfold = X_train[best_selected_features_kfold]
    X_test_best_kfold = X_test[best_selected_features_kfold]

    final_model_kfold = LinearRegression()
    final_model_kfold.fit(X_train_best_kfold, y_train)
    y_pred_test_kfold = final_model_kfold.predict(X_test_best_kfold)

    # Calculate final metrics
    final_rmse_kfold = np.sqrt(mean_squared_error(y_test, y_pred_test_kfold))
    final_r2_kfold = r2_score(y_test, y_pred_test_kfold)

    # Display results
    print("\n----- Final Model Evaluation on Test Set -----")
    print(f"Selected Features: {best_selected_features_kfold}")
    print(f"Final Model RMSE: {final_rmse_kfold:.4f}")
    print(f"Final Model R²:   {final_r2_kfold:.4f}")

    # Display feature selection results
    forward_results_kfold_df
    
    feature_subsets = [
    list(range(20)), #0 to 19 features
    [0],#TODO Try sqrt, sq, cube etc,
    [0, 13, 6, 3, 9]
]


    # Your target column
    target_col = 'Average Difficulty'

    # We'll do a single train/test split once at the beginning
    df_train, df_test = train_test_split(Q9dfgreater10, 
                                        test_size=0.2, 
                                        random_state=RANDOM_SEED)

    # We'll store final results for each subset in a list or dictionary
    all_subsets_results = []

    for idx, subset_cols in enumerate(feature_subsets, start=1):
        print(f"--- Subset #{idx}: {subset_cols} ---")
        
        # 1) Create X_train, y_train, X_test, y_test
        X_train = df_train[subset_cols].to_numpy()
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[subset_cols].to_numpy()
        y_test = df_test[target_col].to_numpy()
        
        # 2) Instantiate RegressionAnalysis for THIS subset
        alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 
                        0.1, 1, 2, 5, 10, 20, 100, 1000, 2000, 10000])
        
        analysis = RegressionAnalysis(
            X_train, y_train,
            alphas=alphas,
            seed=RANDOM_SEED,
            n_splits=5  # K-folds
        )
        
        
        # 3) Cross-validate on the TRAIN set
        analysis.cross_validate()
        
        analysis.plot_cv_rmse_r2()

        # 4) Pick best alpha for Ridge and Lasso based on RMSE from CV
        best_ridge_alpha, best_ridge_rmse = analysis.pick_best_alpha('Ridge', metric='RMSE')
        best_lasso_alpha, best_lasso_rmse = analysis.pick_best_alpha('Lasso', metric='RMSE')
        
        print(f"Best Ridge alpha (subset #{idx}): {best_ridge_alpha} with mean CV-RMSE={best_ridge_rmse:.3f}")
        print(f"Best Lasso alpha (subset #{idx}): {best_lasso_alpha} with mean CV-RMSE={best_lasso_rmse:.3f}")
        
        # 5) Train final models on the full TRAIN set, evaluate once on TEST
        #    This will store results in analysis.results_test
        analysis.finalize_and_evaluate(
            X_train, y_train,
            X_test,  y_test,
            best_ridge_alpha,
            best_lasso_alpha,
            make_residual_plots=False  # Set to True if you want residual subplots
        )
        
        final_test_df = analysis.get_test_results_df()
        print("Final Test Results:\n", final_test_df)
        
        # 6) Optionally, plot coefficients for each final model
        #    a) Normal
        normal_row = final_test_df[final_test_df['Model'] == 'Normal'].iloc[0]
        betas_normal = normal_row['Betas']
        analysis.plot_coefs(
            betas_normal[1:], 
            feature_names=subset_cols,
            model_name=f'Normal (Subset #{idx})'
        )
        
        #    b) Best Ridge
        ridge_row = final_test_df[
            (final_test_df['Model'] == 'Ridge') & 
            (final_test_df['Alpha'] == best_ridge_alpha)
        ].iloc[0]
        betas_ridge = ridge_row['Betas']
        analysis.plot_coefs(
            betas_ridge[1:], 
            feature_names=subset_cols,
            model_name=f'Ridge (Subset #{idx})', 
            alpha=best_ridge_alpha
        )
        
        #    c) Best Lasso
        lasso_row = final_test_df[
            (final_test_df['Model'] == 'Lasso') & 
            (final_test_df['Alpha'] == best_lasso_alpha)
        ].iloc[0]
        betas_lasso = lasso_row['Betas']
        analysis.plot_coefs(
            betas_lasso[1:], 
            feature_names=subset_cols,
            model_name=f'Lasso (Subset #{idx})', 
            alpha=best_lasso_alpha
        )
        
        # Save or store this subset's final results for later
        all_subsets_results.append({
            'subset_index': idx,
            'subset_columns': subset_cols,
            'cv_results': analysis.get_cv_results_df(),  # Cross-validation data
            'test_results': final_test_df                # Final test results
        })
        
        print("------------------------------------------------------\n")

    # After the loop, you can compare all_subsets_results across your feature subsets
    print("All subsets processed and evaluated!")
    print("Question 10")
    print("------------------------------------")
    # 1. Instantiate the class with your dataframes (df_capstone, tagsdf)
#    Suppose df_capstone_dropped_final = ... and tagsdf = ...
    analysis = PepperAnalysis(df_capstone_greater_than_10, tagsdf, seed=RANDOM_SEED)

    # 2. Preprocess data (merge, drop NAs, compute proportions, filter for male/female, etc.)
    analysis.preprocess_data()

    # 3. (Optional) Plot correlation matrix to see the big picture
    analysis.plot_correlation_matrix()

    # 4. (Optional) Plot scatter for single variable
    analysis.plot_scatter_single(x_col='AverageProfessorRating', y_col='Received a pepper')

    # 5. Single-variable logistic regression
    #    You can supply a threshold if you want something other than 0.5
    analysis.logistic_regression_single_var(
        x_col='AverageProfessorRating', 
        y_col='Received a pepper', 
        threshold=0.607  # or pick from ROC
    )

    # 6. Multi-variable logistic regression (by default, it will drop certain columns)
    #    Adjust threshold as needed, or see the "optimal" from the ROC curve
    analysis.logistic_regression_multi_var(threshold=0.465)

    # 7. Train a linear SVM for comparison
    analysis.train_svm()
    
    print('-----------------------------------------------------------------------------------------------------------------')
    print('Question 11')


    df_qual = pd.read_csv('./rmpCapstoneQual.csv', header=None)
    df_qual.columns = ['Major', 'University', 'US State']

    df_merged_qual = pd.concat([df_capstone.drop(columns=['Proportion of students that said they would take the class again', 
                       'Number of ratings coming from online classes']), df_qual], axis=1)

    df_merged_qual_min_10 = df_merged_qual[(df_merged_qual['NumberOfRatings'] >= 10) & ~((df_merged_qual['HighConfMale'] == 1) & (df_merged_qual['HighConfFemale'] == 1)) & ~((df_merged_qual['HighConfMale'] == 0) & (df_merged_qual['HighConfFemale'] == 0))]

    df_merged_qual_min_10 = df_merged_qual_min_10.dropna()

    print(df_merged_qual_min_10.isna().sum())

    visualize_density_plot(df_merged_qual_min_10[df_merged_qual_min_10['US State'] == 'NY'], df_merged_qual_min_10[df_merged_qual_min_10['US State'] == 'NJ'], 'AverageProfessorRating', 'NY', 'NJ')

    perform_ks_mw_test(df_merged_qual_min_10[df_merged_qual_min_10['US State'] == 'NY'], df_merged_qual_min_10[df_merged_qual_min_10['US State'] == 'NJ'], 'AverageProfessorRating', 'NY', 'NJ')


    visualize_density_plot(df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Computer Science'], df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Mathematics'], 'Average Difficulty', 'Computer Science', 'Mathematics')

    perform_ks_mw_test(df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Computer Science'], df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Mathematics'], 'Average Difficulty', 'Computer Science', 'Mathematics')

    visualize_density_plot(df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Communication'], df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Mathematics'], 'Average Difficulty', 'Communication', 'Mathematics')

    perform_ks_mw_test(df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Communication'], df_merged_qual_min_10[df_merged_qual_min_10['Major'] == 'Mathematics'], 'Average Difficulty', 'Communication', 'Mathematics')
    print('-----------------------------------------------------------------------------------------------------------------')
    

            
if __name__ == '__main__':
    main()