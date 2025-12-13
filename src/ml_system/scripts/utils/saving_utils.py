from typing import Any
import polars as pl
import joblib

def save_(to_store: Any, path: str, format: str):
    if format == 'parquet':
        if isinstance(to_store, pl.DataFrame):
            to_store.write_parquet(path)
        else:
            raise AttributeError('This is not a Polars Dataframe')
    elif format == "model":
        try:
            joblib.dump(to_store, path)
        except:
            raise ValueError('Model or preprocessor seems to be invalid')