from ml_system.scripts.utils.loading_utils import load_
from ml_system.scripts.utils.saving_utils import save_
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from omegaconf import DictConfig
import dagshub
import mlflow
from mlflow.models import infer_signature
import polars as pl
from typing import Any
from dotenv import load_dotenv
import os
load_dotenv()


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
    X_train = load_(path=configs.data.paths.X_train, format='parquet')
    X_test = load_(path=configs.data.paths.X_test, format='parquet')
    y_train = load_(path=configs.data.paths.y_train, format='parquet')
    y_test = load_(path=configs.data.paths.y_test, format='parquet')
    y_train_encoded = encode_target(y_train, target)
    y_test_encoded = encode_target(y_test, target)
    full_model_pipeline.fit(X_train, y_train_encoded)
    train_pred = full_model_pipeline.predict(X_train)
    test_pred = full_model_pipeline.predict(X_test)
    save_(to_store=full_model_pipeline, path=configs.features.paths.model, format='model')
    print('Model Trained...')
    make_experiment(configs=configs, assets=[X_train, X_test, y_train_encoded, y_test_encoded, train_pred, test_pred], model=full_model_pipeline)

def make_experiment(configs: DictConfig, assets: list[Any], model: Any):
    X_train, X_test, y_train, y_test, train_pred, test_pred = assets
    dagshub.init(repo_owner='Hg03', repo_name='ml-system', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/Hg03/ml-system.mlflow')
    dagshub.auth.add_app_token(os.getenv('DAGSHUB_USER_TOKEN'))
    with mlflow.start_run():
        mlflow.log_metric('precision', precision_score(y_test, test_pred, average='macro'))
        mlflow.log_metric('recall', recall_score(y_test, test_pred, average='macro'))
        mlflow.log_metric('f1', f1_score(y_test, test_pred, average='macro'))
        signature = infer_signature(X_train.to_pandas(), train_pred)
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path=configs.models.paths.artifacts, 
            signature=signature,
            input_example=X_train.to_pandas()[:5]
        )
    print('Experimentation Done...')