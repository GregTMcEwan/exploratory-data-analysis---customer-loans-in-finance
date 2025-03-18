
# Exploratory Data Analysis - Customer Loans in Finance

## Table of Contents

1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [File Structure](#file-structure)
5. [License Information](#license-information)

## Project Description

### Overview
The **Customer Loans in Finance** project focuses on performing exploratory data analysis (EDA) on a financial institution’s loan portfolio. The goal of this analysis is to provide actionable insights that can aid the institution in making data-driven decisions regarding loan approvals, pricing strategies, and risk management.

### Aim of the Project
The primary aim of this project is to identify patterns, relationships, and anomalies within the loan data to improve the management of risk and return. By thoroughly understanding key characteristics of the loan portfolio—such as risk levels, repayment behavior, and loan distribution—the institution can optimize its lending strategies and enhance profitability.

### What We Learned

Throughout the project, we gained valuable skills and insights that are essential for effective exploratory data analysis (EDA) and decision-making in the financial sector. Key takeaways include:

- **Data Cleaning & Preprocessing**: One of the first and most important steps in any EDA project is to prepare the data for analysis. We learned to handle missing values, clean and format raw data, and preprocess it for further analysis. Techniques such as filling missing values, encoding categorical variables, and standardizing numerical features were applied.

- **Statistical Summary & Descriptive Statistics**: We utilized summary statistics (mean, median, mode, standard deviation, percentiles) to gain a deeper understanding of the distribution and central tendencies within the loan data. This analysis helped identify underlying patterns and outliers that warranted further exploration.

- **Feature Engineering & Transformation**: By creating new features derived from the existing data, we enhanced the predictive power of our models and uncovered more insightful relationships. This included normalizing features and combining variables to capture more meaningful patterns in the data.

- **Data Visualization**: Visualization is an essential tool in EDA for understanding relationships between variables. We used various visualization techniques, such as histograms, scatter plots, boxplots, and bar charts, to explore trends in the loan portfolio. These visualizations highlighted key correlations between loan amounts, interest rates, repayment behavior, and other critical metrics.

- **Correlation & Causation**: We explored the relationships between variables, such as the correlation between loan types and default rates, or the impact of interest rates on repayment terms. This allowed us to identify key factors influencing loan performance and to provide insights for strategic decision-making.

- **Anomaly Detection & Outlier Identification**: We applied statistical techniques and visual methods to detect outliers and anomalies in the data. Identifying unusual patterns or errors was critical for ensuring the integrity of the analysis and for flagging potential risks.

- **Risk Assessment & Decision-Making**: With a deeper understanding of the loan portfolio, we were able to assess the risk associated with different loans. This insight enabled us to propose improvements to the loan approval process and optimize pricing strategies to better manage financial risks.

- **Pattern Recognition & Predictive Insights**: By analyzing various loan attributes, we were able to identify emerging trends and patterns that could inform future loan performance. This provided stakeholders with data-driven insights for making more accurate predictions about the loan portfolio’s behavior.


## Installation Instructions

To get started with the project, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/GregTMcEwan/exploratory-data-analysis---customer-loans-in-finance616.git
   ```

2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```

3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
   The necessary dependencies include:
   - `pandas` (for data manipulation)
   - `sqlalchemy` (for database connection)
   - `plotly` (for data visualization)
   - `pyyaml` (for reading credential files)
   - `scipy` (for statistical analysis)
   - `numpy` (for numerical operations)

## Usage Instructions

Once the dependencies are installed, you can run the Jupyter notebook (`EDA.ipynb`) to perform exploratory data analysis on the loan portfolio dataset. This notebook includes the necessary steps for loading the data, performing analysis, and generating visualizations.

## File Structure

Here is an overview of the project’s directory structure:

```
exploratory-data-analysis---customer-loans-in-finance616/
│
├── db_utils.py               # Database utilities for connecting to RDS and executing queries
├── df_tools.py               # DataFrame tools for data manipulation and cleaning
├── EDA.ipynb                 # Jupyter notebook for exploratory data analysis
├── loan_payments.csv         # Sample loan portfolio data
├── requirements.txt          # Project requirements
├── README.md                 # Project documentation
└── .gitignore                # Files and directories to ignore in version control
```

## License Information

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
