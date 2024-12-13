import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re

class Plotter:
    def __init__(self, data):
        self.data = data



    def pie_chart(self, data, category_col, title='Loan Pie Chart', labels=None):
        """
        Creates a Plotly Express pie chart

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            category_col (str): Column name for the categories to be visualized.
            title (str, optional): The title of the pie chart.
            labels (dict, optional): Labels for the categories.

        Returns:
            fig: A Plotly Express pie chart.
        """
        # Generate pie chart
        fig = px.pie(data_frame=data, names=category_col, title=title, labels=labels)

        # Update layout for better visuals
        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=True
        )
        # Show pie chart
        fig.show()

    def bar_chart(self,
        data, 
        category_col, 
        group_col, 
        agg_col=None, 
        agg_func='count', 
        percentage=False, 
        title='Bar Chart', 
        xlabel=None, 
        ylabel=None
    ):
        """
        Creates a Plotly Express bar chart

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            category_col (str): Column name representing the categories for the x-axis.
            group_col (str): Column name for grouping 
            agg_col (str, optional): Column to aggregate on. If None, aggregation will count rows.
            agg_func (str, optional): Aggregation function ('count', 'sum', 'mean', etc.).
            percentage (bool, optional): Show data as percentages of the total within each category.
            title (str, optional): Chart title.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.

        Returns:
            fig: A Plotly Express bar chart.
        """
        # Aggregate the data
        if agg_func == 'count' and agg_col is None:
            grouped = data.groupby([category_col, group_col]).size().reset_index(name='value')
        else:
            grouped = data.groupby([category_col, group_col])[agg_col].agg(agg_func).reset_index(name='value')
        
        # Convert to percentage if required
        if percentage:
            grouped['percentage'] = grouped.groupby(category_col)['value'].transform(lambda x: round((x / x.sum()) * 100, 2))
            y_col = 'percentage'
            ylabel = ylabel or 'Percentage (%)'
        else:
            y_col = 'value'
            ylabel = ylabel or 'Count'

        # Create the bar chart
        fig = px.bar(
            grouped, 
            x=category_col, 
            y=y_col, 
            color=group_col, 
            barmode='stack',
            title=title,
            labels={category_col: xlabel or category_col, y_col: ylabel, group_col: group_col}
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or category_col,
            yaxis_title=ylabel,
            legend_title=group_col,
            template='plotly_white'
        )
        fig.show()



    def boxplot(self,
            data,
            columns,
            title = 'Loan Information Boxplots'
        ):
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
            title_text=title,
            showlegend=False  # Hide legends for clarity
        )
        fig.show()


    def scattergraph(self,
                data,
                x_col,
                y_col,
                sample_size = 1000,
                marginals = True,
                title = f'Loan DataFrame Scattergraph'

                ):
        # Get a random sample of the DataFrame to reduce visual clutter in the scattergraph
        if len(data) > sample_size:
            data = data.sample(sample_size)
        if marginals == True:
            marg_x = 'violin'
            marg_y = 'box'
        else:
            marg_y = None   
            marg_x = None 

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            marginal_y=marg_x,
            marginal_x=marg_y,
            trendline='ols',
            title=title
        )


        # Display the figure
        fig.show()
    def histogram(self,
                data,
                columns,
                bins=40
                ):
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


    def to_float(self, column):
        # Apply the to_float transformation to a column
        self.data[column] = self.data[column].apply(lambda x: float(re.findall(r'\d+', x)[0]) if isinstance(x, str) else None)

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
    

    def identify_highly_correlated(self, data, threshold=0.8):
        # Compute the correlation matrix
        corr_matrix = data.corr()
        
        # Create an empty list to store columns with high correlation
        highly_correlated = []
        # Loop through the correlation matrix to find highly correlated columns
        for col in corr_matrix.columns:
            for row in corr_matrix.index:
                # Skip the diagonal
                if col != row and corr_matrix.loc[row, col] > threshold:
                    if col not in highly_correlated and row not in highly_correlated:
                        highly_correlated.append(col)  # Mark one of the columns for removal

        return highly_correlated