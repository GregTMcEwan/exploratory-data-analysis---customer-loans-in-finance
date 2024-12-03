

from sqlalchemy import create_engine
import yaml
import pandas as pd

# Class for connecting to and interacting with a remote RDS database
class RDSDatabaseConnector:
    """
    A class for interacting with a remote RDS database.

    This class provides methods for:
    - Establishing a connection to the database using SQLAlchemy.
    - Executing SQL queries and fetching data as Pandas DataFrames.
    - Saving DataFrames as tables in the database.
S
    Attributes:
        engine: The SQLAlchemy engine object for database interactions.

    Methods:
        initialize_sqlachemy(credentials):
            Initializes the SQLAlchemy engine using the provided credentials.

        extract_data(query):
            Extracts data from the database using the specified SQL query.

        save_data(df, table_name):
            Saves a Pandas DataFrame to the database as a new table.
    """

    def __init__(self, credentials):
        # Initialize the SQLAlchemy engine using the provided credentials
        self.engine = self.initialize_sqlachemy(credentials)

    def initialize_sqlachemy(self, credentials):
        try:
            # Create an SQLAlchemy engine to connect to the database
            engine = create_engine(f'postgresql+psycopg2://{credentials["RDS_USER"]}:{credentials["RDS_PASSWORD"]}@{credentials["RDS_HOST"]}:{credentials["RDS_PORT"]}/{credentials["RDS_DATABASE"]}')
            print(f'Connected to remote database')
            return engine
        except Exception as e:
            # Raise an exception if the connection fails
            raise Exception(f"Failed to connect to the remote database: {str(e)}")

    def extract_data(self, query):
        """Extracts data from the database using the given SQL query.

        Args:
            query: The SQL query to execute.

        Returns:
            pandas.DataFrame: The extracted data as a DataFrame.
        """
        with self.engine.connect() as conn:
            # Execute the query and return the results as a DataFrame
            df = pd.read_sql_query(query, conn)
        return df

    def save_data(self, df, table_name):
        """Saves a DataFrame to the database as a new table.

        Args:
            df: The DataFrame to save.
            table_name: The name of the table to create.
        """
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)

# Function to load credentials from a YAML file
    def load_credential_file():
        with open(f'credentials.yaml',) as f:
            credentials = yaml.safe_load(f)
        return credentials

# Function to save a DataFrame to a CSV file
    def save_data_to_file(data_df):
        data_df.to_csv('loan_payments')

class test_class:

    def __init__(self, number):
        self.number = number
    def add(number):
        print(number + number)


