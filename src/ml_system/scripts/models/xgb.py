from ml_system.scripts.utils.loading_utils import load_
from ml_system.scripts.utils.saving_utils import save_
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from omegaconf import DictConfig
import polars as pl
from typing import Any


def get_model(configs: DictConfig):
    model_name = configs.models.active_model
    if model_name == 'xgb':
        params = configs.models.xgb.params
        model = XGBClassifier(
            use_label_encoder=params.use_label_encoder, 
            eval_metric=params.eval_metric, 
            random_state=params.random_state
            )
        return model

def get_full_model_pipeline(preprocessor: Any, model: Any):
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]).set_output(transform='polars')

def encode_target(df: pl.DataFrame, target: str):
    df = df.with_columns(pl.col(target).replace({'At_Risk': 2, 'Stressed': 1, 'Healthy': 0}).cast(pl.Int64))
    return df

def train_model(configs: DictConfig):
    train_preprocessed_data_path = configs.data.paths.train_processed_data
    test_preprocessed_data_path = configs.data.paths.test_processed_data
    preprocessor_path = configs.features.paths.preprocessor
    target = configs.data.columns.target
    model = get_model(configs=configs)
    preprocessor = load_(path=preprocessor_path, format='model')
    full_model_pipeline = get_full_model_pipeline(preprocessor, model)
    train, test = load_(path=train_preprocessed_data_path, format='parquet'), load_(path=test_preprocessed_data_path, format='parquet')
    train, test = encode_target(train, target), encode_target(test, target)
    model.fit(train.drop(target), train.select(pl.col(target)))
    train_pred = model.predict(train.drop(target))
    test_pred = model.predict(test.drop(target))
