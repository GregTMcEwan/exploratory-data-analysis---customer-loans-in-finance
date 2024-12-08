import pandas as pd
import numpy as np
import re

class DataTransform:
    """
    A class for performing transformation operations on a DataFrame

    Attributes:
    df: The DataFrame to perform operations on

    Methods: 

    to_datetime:

    to_float:


    
    """

    def __init__(self, df):
        self.df = df

    def to_datetime(df, format, columns):
        df[columns] = df[columns].apply(lambda col: pd.to_datetime(col, format=format))
        return df
    
    def to_float(x):
        try:
            return float(re.findall(r'\d+', x)[0])
        except:
            return None

