import pandas as pd
import numpy as np



class DataFrameInfo:
    """
    A class for 

    This class provides methods for:
    - Describing all columns in a DataFrame
    - Extracting the median, mean and standard deviation from all columns in the DataFrame
    - Counting distinct values in categorical data
    - Displaying the shape of the DataFrame
    - Generation a percentage count of NULL values of each column
S
    Attributes:
        df: The DataFrame to perform operations on

    Methods:
    TODO

    """

    def __init__(self, df):
        self.df = df

    # Describes all columns in the dataframe
    def describe(self, df):
        print("\n Database Description")
        df.info()

    def stats(self, df):
        print("\n Stats Summary: ")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                print(f"\nColumn: {col}")
                print(f"Median: {round(df[col].median(), 3)}")
                print(f"Standard Deviation: {round(df[col].std(), 3)}")
                print(f"Mean: {round(df[col].mean(), 3)}")
            else:
                print(f"Non numeric column: {df[col]} ")

    def distinct(self, df):
        print("\n Distinct Values: ")
        for col in df.select_dtypes(include=['object', 'category']).columns:
            print(f"\nColumn: {col}")
            print(df[col].value_counts())

    def shape(self, df):
        print(f"\nDataFrame shape: {df.shape}")

    def null_count(self, df):
          print("\n Null Value Counts: ")
          null_counts = df.isnull().sum()
          null_percentages = (null_counts / len(df)) * 100
          print(f"Null Value Counts: {null_counts}")
          print(f"Null Value Percentages: {null_percentages}")       


        

    


