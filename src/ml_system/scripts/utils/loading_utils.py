from omegaconf import DictConfig
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

def load_with_proper_sequence(configs: DictConfig):
    raw_sequence = configs.features.raw_sequence_of_columns
    raw_data_path = configs.data.paths.raw_data
    target_col = configs.data.columns.target
    

    df = pl.read_parquet(raw_data_path)
    return df.select(pl.col(raw_sequence))



