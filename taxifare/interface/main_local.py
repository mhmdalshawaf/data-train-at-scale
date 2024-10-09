import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from taxifare.params import *
from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, save_results, load_model
from taxifare.ml_logic.model import compile_model, initialize_model, train_model
import logging
import traceback
import sys
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO)

def preprocess_and_train(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    GCP_PROJECT_WAGON = "wagon-public-datasets"
    logging.info("Using project: %s", GCP_PROJECT)

    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}`
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()
    credentials = service_account.Credentials.from_service_account_file('/home/mohammed/gcp/macro-shore-437913-t1-0904c6948be4.json')

    if data_query_cached_exists:
        logging.info("Loading data from local CSV...")
        data = pd.read_csv(data_query_cache_path)
    else:
        logging.info("Loading data from BigQuery server...")
        try:
            Client = bigquery.Client(project=GCP_PROJECT,credentials=credentials)
            query_job = Client.query(query)
            result = query_job.result()
            data = result.to_dataframe()
            data.to_csv(data_query_cache_path, header=True, index=False)
        except Exception as e:
            logging.error("Error querying BigQuery: %s", e)
            raise

    data = clean_data(data)

    split_ratio = 0.02
    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1)
    data_val = data.iloc[train_length:, :].sample(frac=1)

    X_train = data_train.drop("fare_amount", axis=1)
    y_train = data_train[["fare_amount"]]
    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    X_train_processed = preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)

    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    model = initialize_model(input_shape=X_train_processed.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    logging.info("✅ preprocess_and_train() done")

def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}`
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        logging.info("Get a DataFrame iterable from local CSV...")
        chunks = pd.read_csv(data_query_cache_path, chunksize=CHUNK_SIZE, parse_dates=["pickup_datetime"])
    else:
        logging.info("Get a DataFrame iterable from querying the BigQuery server...")
        try:
            client = bigquery.Client(project=GCP_PROJECT)
            query_job = client.query(query)
            result = query_job.result(page_size=CHUNK_SIZE)
            chunks = result.to_dataframe_iterable()
        except Exception as e:
            logging.error("Error querying BigQuery: %s", e)
            raise

    for chunk_id, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {chunk_id}...")
        chunk_clean = clean_data(chunk)
        X_chunk = chunk_clean.drop("fare_amount", axis=1)
        y_chunk = chunk_clean[["fare_amount"]]
        X_processed_chunk = preprocess_features(X_chunk)

        chunk_processed = pd.DataFrame(np.concatenate((X_processed_chunk, y_chunk), axis=1))

        chunk_processed.to_csv(
            data_processed_path,
            mode="w" if chunk_id == 0 else "a",
            header=False,
            index=False,
        )

        if not data_query_cache_exists:
            chunk.to_csv(
                data_query_cache_path,
                mode="w" if chunk_id == 0 else "a",
                header=True if chunk_id == 0 else False,
                index=False
            )

    logging.info(f"✅ Data query saved as {data_query_cache_path}")
    logging.info("✅ preprocess() done")

def train(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: train in batches" + Style.RESET_ALL)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []

    chunks = pd.read_csv(data_processed_path, chunksize=CHUNK_SIZE, header=None, dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        logging.info(f"Training on preprocessed chunk n°{chunk_id}")

        learning_rate = 0.0005
        batch_size = 256
        patience = 2
        split_ratio = 0.4

        train_length = int(len(chunk) * (1 - split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        if model is None:
            model = initialize_model(input_shape=X_train_chunk.shape[1:])

        model = compile_model(model, learning_rate)

        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_chunk, y_val_chunk)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)

        logging.info(f"Validation MAE for chunk {chunk_id}: {metrics_val_chunk}")

    val_mae = metrics_val_list[-1]

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    logging.info(f"✅ Trained with MAE: {round(val_mae, 2)}")

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    logging.info("✅ train() done")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    logging.info("✅ pred() done")

    return y_pred

if __name__ == '__main__':
    try:
        preprocess_and_train()
        # preprocess()
        # train() pred()
    except:
  #      import sys
      #  import traceback

  #      import ipdb
      #  extype, value, tb = sys.exc_info()
      #  traceback.print_exc()
      #  ipdb.post_mortem(tb)
        pass
