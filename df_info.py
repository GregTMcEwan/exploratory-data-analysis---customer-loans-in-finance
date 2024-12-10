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

        print("\n Database Info")
        df.info()

    def stats(self, df):

        print("\nStats Summary: ")

        for col in df.columns:

            if df[col].dtype in ['int64', 'float64']:

                print(f"\nColumn: {col}")
                print(f"Mean: {round(df[col].mean(), 3)}")
                print(f"Median: {round(df[col].median(), 3)}")
                print(f"Mode: {df[col].mode()}")
                print(f"Standard Deviation: {round(df[col].std(), 3)}")
                
            else:
                print(f"\nColumn: {col}")
                print(f"Mode: {df[col].mode()}")

    def distinct(self, df):
        print("\nDistinct Values: ")

        for col in df.select_dtypes(include=['object', 'category']).columns:

            print(f"\nColumn: {col}")
            print(df[col].value_counts())

    def shape(self, df):

        print(f"\nDataFrame shape: {df.shape}")

    def null_count(self, df):

        print("\nNull Values: ")

        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)) * 100

        null_percentages = null_percentages[null_percentages>0]
        null_counts = null_counts[null_counts>0]

        if null_counts.empty:
            print("No null values detected")

        else:
            print(f"\nNull Value Counts:\n{null_counts}")
            print(f"\nNull Value Percentages:\n{null_percentages}")

    def duplicates_count(self, df):
        print("\nDuplicate Values: ")
        duplicate_count = df.duplicated().sum()

        print(f"Number of duplicate rows: {duplicate_count}")



        

    


