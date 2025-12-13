from typing import Any
import joblib

def save_(to_store: Any, path: str, format: str):
    if format == 'parquet':
        to_store.write_parquet(path)
    elif format == "model":
        joblib.dump(to_store, path)