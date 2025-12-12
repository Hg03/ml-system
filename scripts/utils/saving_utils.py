from typing import Any
import joblib


def save_(to_store: Any, path: str, format: str):
    if format == "parquet":
        # For dataframe
        to_store.write_parquet(path)
    if format == "model":
        # For models
        joblib.dump(to_store, path)
