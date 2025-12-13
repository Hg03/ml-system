from ml_system.scripts.utils.saving_utils import save_
from omegaconf import DictConfig
import polars as pl


def format_columns(df: pl.DataFrame):
    return df.rename({col: col.lower() for col in df.columns})


def fetch_raw_data(configs: DictConfig):
    data_url = configs.data.url
    raw_data_path = configs.data.paths.raw_data
    df = pl.read_csv(data_url)
    df = format_columns(df)
    save_(to_store=df, path=raw_data_path, format="parquet")
