import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    # YOUR CODE HERE

    # Remove buggy transactions
    # YOUR CODE HERE

    # Remove geographically irrelevant transactions (rows)
    # YOUR CODE HERE

    print("✅ data cleaned")

    return df
