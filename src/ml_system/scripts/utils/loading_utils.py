import polars as pl
from typing import Any
import joblib

def load_(path: str, format: str) -> Any:
    try:
        if format == 'parquet':
            return pl.read_parquet(path)
        elif format == 'model':
            return joblib.load(path)
    except Exception as e:
        raise FileNotFoundError(f"Didn't able to find the location or file for {format} specified at {path}")

