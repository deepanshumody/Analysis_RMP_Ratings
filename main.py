# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

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






def main():
    print("Question 1")
    df_capstone = pd.read_csv('./rmpCapstoneNum.csv', header=0)
    df_capstone.columns = ['AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper', 
                       'Proportion of students that said they would take the class again', 
                       'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale']
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

    


if __name__ == '__main__':
    main()