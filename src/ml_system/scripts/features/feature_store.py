from ml_system.scripts.utils.loading_utils import load_
from dotenv import load_dotenv
from typing import Any
from omegaconf import DictConfig
import polars as pl
import hopsworks
import os

load_dotenv()

def init_hops():
    hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
    return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

def get_fs(project: Any):
    return project.get_feature_store()

def get_fg(fs: Any, configs: DictConfig, split_type: str):
    fg_info = configs.features.store.train if split_type == 'train' else configs.features.store.test
    fg = fs.get_or_create_feature_group(
        name=fg_info.name, 
        version=fg_info.version, 
        description=fg_info.description,
        primary_key=fg_info.primary_key,
        online_enabled=True
    )
    return fg

def add_feature_descriptions(fg, configs: DictConfig):
    for desc in configs.features.store.feature_descriptions:
        fg.update_feature_description(desc["name"], desc["description"])

def add_primary_key(df: pl.DataFrame):
    ids = list(range(len(df)))
    id_series = pl.Series(name='id', values=ids)
    return df.with_columns(id_series)

def push_to_hops(configs: DictConfig):
    if configs.pipeline.type == 'online':
        train = load_(path=configs.data.paths.train_processed_data, format='parquet')
        test = load_(path=configs.data.paths.test_processed_data, format='parquet')
        train, test = add_primary_key(train), add_primary_key(test)

        fs = get_fs(init_hops())
        train_fg, test_fg = get_fg(fs=fs, configs=configs, split_type='train'), get_fg(fs=fs, configs=configs, split_type='test')
        train_fg.insert(train.to_pandas())
        test_fg.insert(test.to_pandas())
        add_feature_descriptions(fg=train_fg, configs=configs)
        add_feature_descriptions(fg=test_fg, configs=configs)
        print('Pushed to Feature Store..')
    else:
        print(f'No feature store, since type is {configs.pipeline.type}')


