import polars as pl
from typing import Any
import joblib

def load_(path: str, format: str) -> Any:
    if format == 'parquet':
        return pl.read_parquet(path)
    elif format == 'model':
        return joblib.load(path)

