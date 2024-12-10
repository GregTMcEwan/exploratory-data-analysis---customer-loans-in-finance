import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Plotter:
    def __init__(self, data):
        self.data = data

    def boxplot(self, data, columns):
        rows = (len(columns) + 1) // 2  # Calculate the number of rows needed
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[
            f"{col}<br><sub>subtitle</sub>" for col in columns
        ])

        for i, col in enumerate(columns):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            # Add the boxplot to the corresponding subplot
            box = go.Box(y=data[col], name=col, boxmean=True)
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            fig.add_trace(box, row=row, col=col_num)

        # Update the layout for better appearance
        fig.update_layout(
            height=300 * rows,  # Adjust height based on number of rows
            title_text="Loan Information Boxplots",
            showlegend=False  # Hide legends for clarity
        )
        fig.show()


    def scattergraph(self, data, x_col, y_col, sample_size = 1000):
        # Get a random sample of the DataFrame to reduce visual clutter in the scattergraph
        if len(data) > sample_size:
            data = data.sample(sample_size)

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            marginal_y='violin',
            marginal_x='box',
            trendline='ols',
            title=f"Scatterplot of {x_col} vs {y_col}"
        )

        fig.update_layout(
            template='plotly_white',
            title=dict(
                text=f"Scatterplot of {x_col} vs {y_col}",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(title=f"{x_col}", showgrid=True),
            yaxis=dict(title=f"{y_col}", showgrid=True)
        )

        # Display the figure
        fig.show()
    def histogram(self, data, columns, bins=40):
        rows = (len(columns) + 1) // 2  # Calculate the number of rows needed
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[
            f"{col}<br><sub>Skew: {round(data[col].skew(), 2)}</sub>" for col in columns
        ])

        for i, col in enumerate(columns):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            # Add the histogram to the corresponding subplot
            hist = go.Histogram(x=data[col], nbinsx=bins, name=col)
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            fig.add_trace(hist, row=row, col=col_num)

        # Update the layout for better appearance
        fig.update_layout(
            height=300 * rows,  # Adjust height based on number of rows
            title_text="Loan Information Histograms",
            showlegend=False  # Hide legends for clarity
        )
        fig.show()
        
    def normal_test(self, columns):

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            stat, p = stats.normaltest(self.data[col], nan_policy='omit')
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

        # Plotting the comparison
        fig = px.bar(
            null_comparison,
            x='Columns',
            y='Null Count',
            color='Stage',
            barmode='group',
            title='Null Value Counts Before and After Handling',
        )

        
        fig.update_layout(
            xaxis_title='Columns',
            yaxis_title='Number of Null Values',
            xaxis_tickangle=-45,
            bargap=0.2,
            width=900,  
            height=500,  
        )

        fig.show()


    def plot_correlation_matrix(self, data):
        corr_matrix = data.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix",
            labels={'x': 'Columns', 'y': 'Columns'}
        )
        fig.update_layout(
        height=1300,  
        width=1300,   
        title_x=0.5,  
        title_y=0.95  
    )
        fig.show()




class DataFrameTransform:
    def __init__(self, data):
        self.data = data

    def update_data(self, data):
        self.data = data

    def to_datetime(self, data, format, columns):
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            data[col] = pd.to_datetime(data[col], format=format, errors='coerce')
        return data
    

    def fill_missing(self, columns, imputation='mode'):
        """
        Impute missing values for one or more columns.
        
        Args:
        - columns: str or list of str, the column(s) to impute.
        - imputation: str, the imputation strategy ('mean', 'median', 'mode', or 'zero').

        Returns:
        - None: Updates the DataFrame in place.
        """
        
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
    
    def reduce_skew(self, columns, method='log'):
        """
        Reduces skewness of the specified columns using a transformation method.
        Args:
            columns (list): List of column names to transform.
            method (str): Transformation method ('log', 'sqrt', 'boxcox').
        Returns:
            DataFrame: Transformed DataFrame.
        """
        for col in columns:
            if method == 'log':
                self.data[col] = np.log1p(self.data[col].clip(lower=0))
            elif method == 'boxcox':
                transformed_data, _ = stats.boxcox(self.data[col].clip(lower=1))  
                self.data[col] = transformed_data
            elif method == 'yeo-johnson':
                transformed_data, _ = stats.yeojohnson(self.data[col])  
                self.data[col] = transformed_data
        return self.data

    def remove_outliers(self, columns, method="iqr", threshold=1.5):
        """
        Removes outliers from numerical columns.
        :param columns: List of columns to process.
        :param method: Outlier detection method ('iqr' or 'zscore').
        :param threshold: Threshold for defining outliers.
        """
        count_before_removal = len(self.data)
        if method == "iqr":
            filters = []
            for col in columns:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                filters.append((self.data[col] >= lower_bound) & (self.data[col] <= upper_bound))
            
            combined_filter = np.logical_and.reduce(filters)
            self.data = self.data[combined_filter]
        
        elif method == "zscore":
            zscores = self.data[columns].apply(stats.zscore).abs()
            self.data = self.data[(zscores <= threshold).all(axis=1)]

        count_after_removal = len(self.data)
        print(f'Outliers removed using {method}. {count_before_removal-count_after_removal} rows removed')
        
        return self.data