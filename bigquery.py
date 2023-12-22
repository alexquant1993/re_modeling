import os

import pandas as pd
import polars as pl
from google.cloud import bigquery


def download_data_from_bigquery_to_local(
    dataset_id: str,
    table_id: str,
    credentials_path: str,
    local_path: str = None,
    format: str = "polars",
) -> pd.DataFrame or pl.DataFrame or None:
    """
    Download data from a BigQuery table to a local file or directly into a DataFrame.

    Args:
        dataset_id: The ID of the BigQuery dataset.
        table_id: The ID of the BigQuery table.
        credentials_path: The path to the GCP credentials file.
        local_path: The path where the local file should be saved.
        format: The format for the file ('csv' or 'pandas'/'polars' for DataFrame).

    Returns:
        Optionally returns a DataFrame if format is 'pandas' or 'polars'.
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(credentials_path)

    client = bigquery.Client()

    # Construct a reference to the dataset
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    # Get the BigQuery table
    table = client.get_table(table_ref)

    # Extract the table to a DataFrame
    print(f"Starting extraction of {table.num_rows} rows...")
    df = client.list_rows(table).to_dataframe()
    print("Extraction complete.")

    if format == "pandas" or format == "polars":
        if format == "polars":
            df = pl.from_pandas(df)
        return df
    elif format == "csv" and local_path is not None:
        os.makedirs(local_path, exist_ok=True)
        destination_uri = f"{local_path}/{table_id}.csv"
        df.to_csv(destination_uri, index=False)
        print(f"Exported {dataset_id}.{table_id} to {destination_uri}")
    else:
        raise ValueError("Invalid format or local_path is None for format 'csv'")
