import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re

class Plotter:
    """
    A utility class for creating various visualizations using Plotly Express and Plotly Graph Objects.

    This class provides methods for:
    - Pie charts
    - Bar charts
    - Boxplots
    - Scatter plots with optional marginals and trendlines

    Attributes:
        data (pd.DataFrame): The DataFrame to visualize.
    """

    def __init__(self, data):
        """
        Initializes the Plotter class with a DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame to be visualized.
        """
        self.data = data

    def histogram(self, data, columns, bins=40):
        """
        Plots histograms for multiple columns in a DataFrame with skewness displayed as subtitles.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing the data to plot.
            columns (list): List of column names for which histograms will be generated.
            bins (int, optional): Number of bins to use in the histograms. Default is 40.

        Raises:
            ValueError: If any column in `columns` is not present in the DataFrame.

        Description:
            - Creates a subplot for each column specified.
            - Displays skewness as part of the subplot title.
            - Arranges plots in a grid of two columns per row.

        Returns:
            None: The function directly displays the histograms using Plotly.
        """
        # Calculate the number of rows required for the subplots
        rows = (len(columns) + 1) // 2

        # Create subplots with titles that include skewness for each column
        fig = make_subplots(
            rows=rows, 
            cols=2, 
            subplot_titles=[
                f"{col}<br><sub>Skew: {round(data[col].skew(), 2)}</sub>" for col in columns
            ]
        )

        # Loop through each column and add histograms to the subplots
        for i, col in enumerate(columns):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            
            # Create a histogram for the column
            hist = go.Histogram(x=data[col], nbinsx=bins, name=col)
            
            # Determine the row and column position for the subplot
            row = (i // 2) + 1
            col_num = (i % 2) + 1

            # Add the histogram to the appropriate subplot
            fig.add_trace(hist, row=row, col=col_num)

        # Update the layout for the plot
        fig.update_layout(
            height=300 * rows,  # Dynamically set height based on the number of rows
            title_text="Loan Information Histograms",
            showlegend=False  # Hide legends for clarity
        )

        # Display the figure
        fig.show()

    def normal_test(self, columns):
        """
        Conducts a normality test for specified columns in the DataFrame.

        Parameters:
            columns (list): List of column names to test for normality.

        Raises:
            ValueError: If any column in `columns` is not present in the DataFrame.

        Description:
            - Uses the D'Agostino and Pearson's test for normality.
            - Prints the test statistic and p-value for each column.

        Returns:
            None: The function outputs the results to the console.
        """
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            
            # Perform the normality test
            stat, p = stats.normaltest(self.data[col], nan_policy='omit')
            
            # Print the test results
            print(f'{col} normal test statistics: {stat:.3f}, p={p:.3f}')

    def plot_correlation_matrix(self, data):
        """
        Plots a heatmap of the correlation matrix for a given DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame for which the correlation matrix is calculated.

        Description:
            - Computes the correlation matrix of the input DataFrame.
            - Visualizes the matrix as a heatmap using Plotly.
            - Colors indicate the strength and direction of correlations.
            - Displays the correlation values within the heatmap for clarity.

        Returns:
            None: The function directly displays the correlation matrix as a heatmap.
        """
        # Compute the correlation matrix for the DataFrame
        corr_matrix = data.corr()

        # Create a heatmap visualization using Plotly
        fig = px.imshow(
            corr_matrix,  # Input the correlation matrix
            text_auto=True,  # Automatically display correlation values in the heatmap
            color_continuous_scale='RdBu_r',  # Set the color scale (red to blue reversed)
            title="Correlation Matrix",  # Set the title of the plot
            labels={'x': 'Columns', 'y': 'Columns'}  # Label axes
        )

        fig.update_layout(
            height=1300,  
            width=1300,   
            title_x=0.5,  
            title_y=0.95  
        )

        fig.show()


    def plot_null_removal(self, original_df, cleaned_df):
        """
        Visualizes the change in null value counts before and after cleaning a DataFrame.

        Parameters:
            original_df (pd.DataFrame): The original DataFrame with null values.
            cleaned_df (pd.DataFrame): The cleaned DataFrame after handling null values.

        Description:
            - Compares the number of null values in each column before and after cleaning.
            - Plots a grouped bar chart showing the null counts for each column.

        Returns:
            None: The function directly displays the bar chart using Plotly.
        """
        # Count null values in the original and cleaned DataFrames
        original_null_count = original_df.isnull().sum()
        clean_null_count = cleaned_df.isnull().sum()

        # Create a comparison DataFrame
        null_comparison = pd.DataFrame({
            'Before': original_null_count,
            'After': clean_null_count
        }).reset_index().rename(columns={'index': 'Columns'})

        # Transform the data for plotting
        null_comparison = null_comparison.melt(
            id_vars='Columns', 
            value_vars=['Before', 'After'], 
            var_name='Stage', 
            value_name='Null Count'
        )

        # Create a bar chart to visualize null counts
        fig = px.bar(
            null_comparison,
            x='Columns',
            y='Null Count',
            color='Stage',
            barmode='group',
            title='Null Value Counts Before and After Handling',
        )

        # Update the layout for better appearance
        fig.update_layout(
            xaxis_title='Columns',
            yaxis_title='Number of Null Values',
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            bargap=0.2,  # Set gap between bars
            width=900,  # Set figure width
            height=500,  # Set figure height
        )

        # Display the figure
        fig.show()


    def pie_chart(self, data, category_col, title='Pie Chart', labels=None):
        """
        Creates a pie chart using Plotly Express.

        Parameters:
            category_col (str): Column name for the categories to be visualized.
            title (str, optional): The title of the pie chart. Default is 'Pie Chart'.
            labels (dict, optional): Custom labels for the categories.

        Returns:
            None: Displays the pie chart.
        """
        fig = px.pie(
            data_frame=data,
            names=category_col,
            title=title,
            labels=labels
        )

        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=True
        )
        fig.show()

    def bar_chart(self, data, category_col, group_col, agg_col=None, agg_func='count', percentage=False, 
                  title='Bar Chart', xlabel=None, ylabel=None):
        """
        Creates a bar chart using Plotly Express.

        Parameters:
            category_col (str): Column name representing the categories for the x-axis.
            group_col (str): Column name for grouping the data.
            agg_col (str, optional): Column to aggregate on. Defaults to None (row count).
            agg_func (str, optional): Aggregation function ('count', 'sum', 'mean', etc.). Default is 'count'.
            percentage (bool, optional): Show data as percentages within each category. Default is False.
            title (str, optional): Title of the bar chart. Default is 'Bar Chart'.
            xlabel (str, optional): Custom label for the x-axis. Default is None.
            ylabel (str, optional): Custom label for the y-axis. Default is None.

        Returns:
            None: Displays the bar chart.
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
        """
        Creates Plotly boxplots using graphobjects for subplots

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            columns (list): Columns to be made into boxplots
            title (str, optional): Title of the plot. Default is 'Loan Information Boxplots'

        Returns:
            None: The function directly displays the boxplots using Plotly.
        """

        rows = (len(columns) + 1) // 2  # Calculate the number of rows needed 
        # Creates the sublot
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[
            f"{col}<br><sub>subtitle</sub>" for col in columns
        ])
        # Creates the boxplots by iterating through the given columns
        for i, col in enumerate(columns):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            # Add the boxplot to the corresponding subplot
            box = go.Box(y=data[col], name=col, boxmean=True)
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            fig.add_trace(box, row=row, col=col_num)

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
                sample_size=1000,
                marginals=True,
                title=f'Loan DataFrame Scattergraph'
                ):
        """
        Creates a Plotly Express scattergraph with optional marginals.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            x_col (str): Column name for the x-axis.
            y_col (str): Column name for the y-axis.
            sample_size (int, optional): The sample size for the scattergraph. Default is 1000.
            marginals (bool, optional): Whether the plot has marginal plots. Default is True.
            title (str, optional): Title of the plot. Default is 'Loan DataFrame Scattergraph'.

        Returns:
            None: Displays the scattergraph using Plotly.
        """
        # Reduce the data to a random sample if its size exceeds the specified sample size
        if len(data) > sample_size:
            data = data.sample(sample_size)
        
        # Determine the type of marginal plots
        if marginals:
            marg_x = 'violin'
            marg_y = 'box'
        else:
            marg_y = None
            marg_x = None

        # Create the scattergraph with optional marginals and regression line
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            marginal_y=marg_x,
            marginal_x=marg_y,
            trendline='ols',  # Add an OLS regression line
            title=title
        )
        # Display the scattergraph
        fig.show()





class DataFrameTransform:
    """
    A utility class for performing various data transformation tasks on a DataFrame.

    Attributes:
        data (pd.DataFrame): The DataFrame to be transformed.

    Methods:
        update_data(data): Updates the class's DataFrame with a new DataFrame.
        to_float(column): Converts a column with numeric strings to floats.
        to_datetime(data, format, columns): Converts specified columns to datetime format.
        fill_missing(columns, imputation): Imputes missing values in specified columns.
        reduce_skew(columns, method): Reduces skewness in specified columns.
        remove_outliers(columns, method, threshold): Removes outliers from specified columns.
        identify_highly_correlated(data, threshold): Identifies highly correlated columns.
    """
    
    def __init__(self, data):
        """
        Initializes the DataFrameTransform class with a DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
        """
        self.data = data

    def update_data(self, data):
        """
        Updates the internal DataFrame.

        Parameters:
            data (pd.DataFrame): The new DataFrame to replace the existing one.
        """
        self.data = data

    def to_float(self, column):
        """
        Converts numeric strings in a specified column to floats.

        Parameters:
            column (str): The column to be transformed.

        Returns:
            None: Updates the DataFrame in place.

        Raises:
            ValueError: If the column does not exist in the DataFrame.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        
        # Apply the transformation, extracting numbers from strings
        self.data[column] = self.data[column].apply(
            lambda x: float(re.findall(r'\d+', x)[0]) if isinstance(x, str) else None
        )

    def to_datetime(self, data, format, columns):
        """
        Converts specified columns to datetime format.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            format (str): The datetime format string.
            columns (list): List of column names to convert.

        Returns:
            pd.DataFrame: The DataFrame with updated columns.

        Raises:
            ValueError: If a specified column does not exist.
        """
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            # Convert the column to datetime, coercing errors to NaT
            data[col] = pd.to_datetime(data[col], format=format, errors='coerce')
        return data

    def fill_missing(self, columns, imputation='mode'):
        """
        Imputes missing values in the specified columns.

        Parameters:
            columns (list): List of column names to impute.
            imputation (str): The imputation strategy ('mean', 'median', 'mode', or 'zero').

        Returns:
            None: Updates the DataFrame in place.

        Raises:
            ValueError: If an invalid imputation strategy is provided or if a column does not exist.
        """
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
                
            # Perform imputation based on the specified strategy
            if imputation == 'mean':
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif imputation == 'median':
                self.data[col] = self.data[col].fillna(self.data[col].median())
            elif imputation == 'mode':
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
            elif imputation == 'zero':
                self.data[col] = self.data[col].fillna(0)
            else:
                raise ValueError("Invalid imputation strategy. Choose 'mean', 'median', 'mode', or 'zero'.")

    def reduce_skew(self, columns, method='log'):
        """
        Reduces skewness of specified columns using a transformation method.

        Parameters:
            columns (list): List of column names to transform.
            method (str): Transformation method ('log', 'sqrt', 'boxcox', 'yeo-johnson').

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            ValueError: If a column does not exist or an invalid method is specified.
        """
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            
            # Apply the selected transformation method
            if method == 'log':
                self.data[col] = np.log1p(self.data[col].clip(lower=0))
            elif method == 'boxcox':
                transformed_data, _ = stats.boxcox(self.data[col].clip(lower=1))
                self.data[col] = transformed_data
            elif method == 'yeo-johnson':
                transformed_data, _ = stats.yeojohnson(self.data[col])
                self.data[col] = transformed_data
            else:
                raise ValueError("Invalid method. Choose 'log', 'boxcox', or 'yeo-johnson'.")
        return self.data

    def remove_outliers(self, columns, method="iqr", threshold=1.5):
        """
        Removes outliers from specified columns.

        Parameters:
            columns (list): List of column names to process.
            method (str): Outlier detection method ('iqr' or 'zscore').
            threshold (float): Threshold for defining outliers.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed.

        Raises:
            ValueError: If a column does not exist or an invalid method is specified.
        """
        count_before_removal = len(self.data)
        
        if method == "iqr":
            filters = []
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
                # Calculate IQR
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
        else:
            raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
        
        count_after_removal = len(self.data)
        print(f'Outliers removed using {method}. {count_before_removal-count_after_removal} rows removed')
        
        return self.data

    def identify_highly_correlated(self, data, threshold=0.8):
        """
        Identifies highly correlated columns based on a threshold.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            threshold (float): Correlation threshold for identifying columns.

        Returns:
            list: List of highly correlated column names.
        """
        corr_matrix = data.corr()
        highly_correlated = []
        
        # Identify pairs of highly correlated columns
        for col in corr_matrix.columns:
            for row in corr_matrix.index:
                if col != row and corr_matrix.loc[row, col] > threshold:
                    if col not in highly_correlated and row not in highly_correlated:
                        highly_correlated.append(col)
        
        return highly_correlated
    


class DataFrameInfo:
    """
    A class for performing various operations on a DataFrame.

    This class provides methods for:
    - Describing all columns in a DataFrame
    - Extracting the median, mean, and standard deviation from all numerical columns
    - Counting distinct values in categorical data
    - Displaying the shape of the DataFrame
    - Generating a percentage count of NULL values for each column

    Attributes:
        df: The DataFrame to perform operations on
    """

    def __init__(self, df):
        """
        Initializes the DataFrameInfo class with the specified DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to be analyzed.
        """
        self.df = df

    # Describes all columns in the DataFrame
    def describe(self, df):
        """
        Prints the basic information about the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to describe.
        """
        print("\n Database Info")
        df.info()

    def stats(self, df):
        """
        Prints statistical summaries (mean, median, mode, standard deviation) for numerical columns
        and the mode for non-numerical columns.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        print("\nStats Summary: ")

        for col in df.columns:  # Iterate through all columns in the DataFrame
            if df[col].dtype in ['int64', 'float64']:  # Check if the column is numerical
                print(f"\nColumn: {col}")
                print(f"Mean: {round(df[col].mean(), 3)}")  # Print mean
                print(f"Median: {round(df[col].median(), 3)}")  # Print median
                print(f"Mode: {df[col].mode()}")  # Print mode
                print(f"Standard Deviation: {round(df[col].std(), 3)}")  # Print standard deviation
            else:  # For non-numerical columns
                print(f"\nColumn: {col}")
                print(f"Mode: {df[col].mode()}")  # Print mode

    def distinct(self, df):
        """
        Prints the count of distinct values for each categorical column in the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        print("\nDistinct Values: ")

        # Select columns with categorical or object data types
        for col in df.select_dtypes(include=['object', 'category']).columns:
            print(f"\nColumn: {col}")
            print(df[col].value_counts())  # Print value counts for each category

    def shape(self, df):
        """
        Prints the shape of the DataFrame (number of rows and columns).

        Parameters:
            df (pd.DataFrame): The DataFrame whose shape is to be printed.
        """
        print(f"\nDataFrame shape: {df.shape}")

    def null_count(self, df):
        """
        Prints the count and percentage of NULL values in each column.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        print("\nNull Values: ")

        null_counts = df.isnull().sum()  # Count NULL values for each column
        null_percentages = (null_counts / len(df)) * 100  # Calculate percentages

        # Filter columns with NULL values
        null_percentages = null_percentages[null_percentages > 0]
        null_counts = null_counts[null_counts > 0]

        if null_counts.empty:  # Check if there are no NULL values
            print("No null values detected")
        else:
            print(f"\nNull Value Counts:\n{null_counts}")  # Print count of NULL values
            print(f"\nNull Value Percentages:\n{null_percentages}")  # Print percentage of NULL values

    def duplicates_count(self, df):
        """
        Prints the count of duplicate rows in the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        print("\nDuplicate Values: ")
        duplicate_count = df.duplicated().sum()  # Count duplicate rows
        print(f"Number of duplicate rows: {duplicate_count}")
