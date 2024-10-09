import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

def clean_data(df: pd.DataFrame, is_chunked: bool = False) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions

    Parameters:
    - df: DataFrame to clean
    - is_chunked: Boolean indicating if the data is processed in chunks
    """

    df = df.astype(DTYPES_RAW)

    df = df.drop_duplicates()

    if not is_chunked:
        df = df.dropna(how='any', axis=0)

    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
            (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]

    df = df[df.passenger_count > 0]
    df = df[df.fare_amount > 0]

    df = df[df.fare_amount < 400]
    df = df[df.passenger_count < 8]

    df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    print("✅ data cleaned")

    return df
