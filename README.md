# Analysis of Rate My Professors (RMP) Ratings

This repository contains an analysis of Rate My Professors (RMP) ratings, conducted to explore patterns, trends, and insights about professor evaluations by students. The project employs rigorous statistical and machine learning techniques to answer various research questions and make informed conclusions about the dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Key Findings](#key-findings)
- [Statistical Tests and Methodologies](#statistical-tests-and-methodologies)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Project Overview

The goal of this project is to analyze trends and biases in RMP ratings. The project includes:

- Hypothesis testing for gender differences in ratings.
- Exploring the impact of confounding variables like "pepper" (a marker for attractiveness) and years of experience.
- Predictive modeling to identify key features influencing ratings.
- Data-driven visualizations for intuitive understanding.

This work combines data preprocessing, exploratory data analysis (EDA), statistical tests, and regression modeling to achieve these goals. It also evaluates the practical implications of findings through power analysis and effect size measurement.

For a detailed report on the analysis and findings, refer to the [Project Report](https://docs.google.com/document/d/1M5GcozvDQxWaWHERhlZjKv4_HsLnH_WzgyD-icaFvdw/edit?usp=sharing).

---

## Features

- **Data Cleaning and Preprocessing**: Removes low-confidence and incomplete data entries, handles missing values, and normalizes data for analysis.
- **Statistical Testing**: Employs Kolmogorov-Smirnov, Mann-Whitney U, Levene's test, Kruskal-Wallis test, and others for hypothesis testing.
- **Predictive Modeling**: Implements Ridge, Lasso, and OLS regression for feature importance and prediction.
- **Power Analysis**: Evaluates the robustness of statistical findings to determine reproducibility and significance.
- **Interactive Visualizations**: Creates plots and charts for insightful analysis of trends, distributions, and patterns.
- **Feature Selection**: Utilizes Lasso regularization and forward feature selection to identify key predictors.

---

## Key Findings

1. **Gender Differences**:
   - Significant differences were observed between male and female professors in specific conditions (e.g., 19+ reviews without a pepper rating).
   - Power analysis suggested low reproducibility, highlighting the need for caution in interpreting results.

2. **Impact of Confounding Variables**:
   - "Pepper" and years of experience significantly influence ratings, necessitating adjustment in analyses.
   - Difficulty ratings and average ratings show minimal gender differences after controlling for confounds.

3. **Predictive Modeling**:
   - The "Proportion of students who would take the class again" was the most predictive feature for average ratings.
   - "Tough Grader" was a key feature for predicting difficulty ratings but demonstrated issues with multicollinearity.

4. **Variance in Ratings**:
   - Variance differences by gender were observed in very specific contexts but were not generalizable.

5. **State-Wise Analysis**:
   - No significant difference in professor ratings was found between NY and NJ despite expectations.

---

## Statistical Tests and Methodologies

1. **Hypothesis Testing**:
   - Kolmogorov-Smirnov and Mann-Whitney U tests were used to compare distributions and medians of ratings.
   - Levene's test assessed variance differences in ratings between genders.

2. **Confounding Variables**:
   - Confounds such as "pepper" (attractiveness), years of experience (proxied by number of ratings), and difficulty ratings were identified and adjusted.

3. **Regression Models**:
   - Ridge and Lasso regression were used for feature selection and regularization.
   - OLS regression provided interpretable models for predictive analysis.

4. **Power Analysis**:
   - Post-hoc effect size calculations and power analysis determined the reliability of significant findings.

5. **Feature Engineering**:
   - New features such as "Proportion of students who would take the class again" and normalized tag columns were introduced to improve model performance.

---

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Samarth26/Analysis_RMP_Ratings.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Analysis_RMP_Ratings
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

4. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Ensure all dependencies are installed.
2. Place your dataset in the `data/` directory.
3. Run the analysis script:

   ```bash
   python analysis_script.py
   ```

4. Explore results in the output folder or through generated plots.

---

## Data Sources

This project uses a cleaned dataset derived from Rate My Professors. Detailed preprocessing steps are documented in the `scripts/` folder and the accompanying capstone report.

---

## Project Structure

```plaintext
Analysis_RMP_Ratings/
│
├── data/               # Contains raw and processed datasets
├── notebooks/          # Jupyter notebooks for exploratory analysis
├── scripts/            # Python scripts for analysis
├── reports/            # Capstone report detailing findings
├── requirements.txt    # List of required dependencies
├── README.md           # Project documentation
└── LICENSE             # License for the repository
```

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, feel free to:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add your message here"
   ```

4. Push the branch:

   ```bash
   git push origin feature-name
   ```

5. Submit a pull request.

---

