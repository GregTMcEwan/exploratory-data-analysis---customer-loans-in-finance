import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import normaltest

class Plotter:
    def __init__(self, data):
        self.data = data

    def histogram(self, columns):

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            hist = px.histogram(self.data[col], x=[col])
            hist.show()
            
    def normal_test(self, columns):

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            stat, p = normaltest(self.data[col], nan_policy='omit')
            print(f'{col} normal test statistics: {stat:.3f}, p={p:.3f}')

    def plot_null_removal(self, original_df, cleaned_df):

        original_null_count = original_df.isnull().sum()
        clean_null_count = cleaned_df.isnull().sum()

        null_comparison = pd.DataFrame({
            'Before': original_null_count,
            'After': clean_null_count
        }).reset_index().rename(columns={'index': 'Columns'})


        null_comparison = null_comparison.melt(id_vars='Columns', 
                                               value_vars=['Before', 'After'], 
                                               var_name='Stage', 
                                               value_name='Null Count')

        # Plot with Plotly Express
        fig = px.bar(
            null_comparison,
            x='Columns',
            y='Null Count',
            color='Stage',
            barmode='group',
            title='Null Value Counts Before and After Handling',
        )

        # Customize layout
        fig.update_layout(
            xaxis_title='Columns',
            yaxis_title='Number of Null Values',
            xaxis_tickangle=-45,
            bargap=0.2,
            width=900,  # Adjust width
            height=500,  # Adjust height
        )

        # Show plot
        fig.show()



class DataFrameTransform:
    def __init__(self, data):
        self.data = data    

    def fill_missing(self, columns, imputation='mode'):
        """
        Impute missing values for one or more columns.
        
        Args:
        - columns: str or list of str, the column(s) to impute.
        - imputation: str, the imputation strategy ('mean', 'median', 'mode', or 'zero').

        Returns:
        - None: Updates the DataFrame in place.
        """
        if isinstance(columns, str):  # Handle single column case
            columns = [columns]
        
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
                
            if imputation == 'mean':
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif imputation == 'median':
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif imputation == 'mode':
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif imputation == 'zero':
                self.data[col].fillna(0, inplace=True)
            else:
                raise ValueError("Invalid imputation strategy. Choose 'mean', 'median', 'mode', or 'zero'.")

    def remove_outliers(self, columns, method="iqr", threshold=1.5):
        """
        Removes outliers from a numerical column.
        :param column: Column to process.
        :param method: Outlier detection method ('iqr' or 'zscore').
        :param threshold: Threshold for defining outliers.
        """
        if method == "iqr":
            q1 = self.data[columns].quantile(0.25)
            q3 = self.data[columns].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            self.data = self.data[(self.data[columns] >= lower_bound) & (self.data[columns] <= upper_bound)]
            
        elif method == "zscore":

            self.data = self.data[(np.abs(zscore(self.data[columns])) <= threshold)]