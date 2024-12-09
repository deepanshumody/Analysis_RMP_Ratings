# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Seed value for random number generators to obtain reproducible results
RANDOM_SEED = 10676128

# Apply the random seed to numpy.
np.random.seed(RANDOM_SEED)

def visualize_density_plot(df1, df2, column, hue=None):
    # Plot the histogram of the AverageProfessorRating for professors with more than 10 ratings for women and men separately
    plt.figure(figsize=(10, 6))

    sns.histplot(df1[column], bins=30, kde=True, color='blue', label='AverageProfessorRating for Men', stat='density')
    sns.histplot(df2[column], bins=30, kde=True, color='red', label='AverageProfessorRating for Women', stat='density')

    plt.title('Normalized Histogram of AverageProfessorRating for Professors with More Than 10 Ratings')
    plt.xlabel('AverageProfessorRating')
    plt.ylabel('Density')
    plt.legend()
    #plt.show()

def perform_ks_mw_test(df1, df2, column, str1, str2):
    # Perform a KS test to check if the distributions of AverageProfessorRating for Male and Felame professors with more than 10 ratings are the same
    ks_stat, p_val = stats.ks_2samp(df1[column], df2[column])
    mannwhitney_stat, mannwhitney_p_val = stats.mannwhitneyu(df1[column], df2[column])
    print(f'KS Test of {column} for the two groups: {str1} and {str2}')
    print('KS Statistic: ', ks_stat)
    print('P-value: ', p_val)
    print('Mann Whitney U Statistic: ', mannwhitney_stat)
    print('Mann Whitney U P-value: ', mannwhitney_p_val)


def main():
    df_capstone = pd.read_csv('./rmpCapstoneNum.csv', header=0)
    df_capstone.columns = ['AverageProfessorRating', 'Average Difficulty', 'NumberOfRatings', 'Received a pepper', 
                       'Proportion of students that said they would take the class again', 
                       'Number of ratings coming from online classes', 'HighConfMale', 'HighConfFemale']
    df_capstone_greater_than_10 = df_capstone[(df_capstone['NumberOfRatings'] >= 10) & ~((df_capstone['HighConfMale'] == 1) & (df_capstone['HighConfFemale'] == 1)) & ~((df_capstone['HighConfMale'] == 0) & (df_capstone['HighConfFemale'] == 0))]
    df_capstone_greater_than_10_men = df_capstone_greater_than_10[df_capstone_greater_than_10['HighConfMale'] == 1]
    df_capstone_greater_than_10_female = df_capstone_greater_than_10[df_capstone_greater_than_10['HighConfFemale'] == 1]
    visualize_density_plot(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating')
    perform_ks_mw_test(df_capstone_greater_than_10_men, df_capstone_greater_than_10_female, 'AverageProfessorRating', "df_capstone_greater_than_10_men", "df_capstone_greater_than_10_female")
    

if __name__ == '__main__':
    main()