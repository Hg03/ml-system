from sklearn.impute import SimpleImputer
from ml_system.scripts.utils.saving_utils import save_
from ml_system.scripts.utils.loading_utils import load_
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig, ListConfig
import polars as pl


def make_imputer(configs: DictConfig):
    numerical_imputer = SimpleImputer(strategy=configs.features.strategy.num_impute)
    numerical_columns = list(configs.data.columns.numerical_features)
    categorical_imputer = SimpleImputer(strategy=configs.features.strategy.cat_impute)
    categorical_columns = list(configs.data.columns.categorical_features)
    return make_column_transformer(
        (numerical_imputer, numerical_columns), 
        (categorical_imputer, categorical_columns), remainder='passthrough').set_output(transform='polars')


def make_encoder(configs: DictConfig):
    encoder = OrdinalEncoder()
    ordinal_columns = list(configs.data.columns.ordinal_features)
    return make_column_transformer(
        (encoder, ordinal_columns),
        remainder='passthrough'
    ).set_output(transform='polars')

def postprocessor(df):
    return df.rename({col: col[col.rfind('__')+2:] for col in df.columns})

def make_preprocessor(configs: DictConfig):
    postprocess = FunctionTransformer(postprocessor)
    return Pipeline(steps=
    [('imputer', make_imputer(configs)), 
    ('postprocess1', postprocess),
    ('encoder', make_encoder(configs)),
    ('postprocess2', postprocess)])

def preprare_to_preprocess(df: pl.DataFrame, configs: DictConfig):
    columns_required = list(configs.data.columns.features) + [configs.data.columns.target]
    return df.select(pl.col(columns_required)), df.select(pl.col(configs.data.columns.features)), df.select(pl.col([configs.data.columns.target]))

def transform_data(configs: DictConfig):
    raw_data_path = configs.data.paths.raw_data
    train_preprocessed_data_path = configs.data.paths.train_processed_data
    test_preprocessed_data_path = configs.data.paths.test_processed_data
    preprocessor_path = configs.features.paths.preprocessor
    test_size = configs.features.test_size
    df = load_(path=raw_data_path, format='parquet')
    df, X, y = preprare_to_preprocess(df=df, configs=configs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    preprocessor = make_preprocessor(configs=configs)
    train_transformed_X = preprocessor.fit_transform(X_train)
    test_transformed_X = preprocessor.transform(X_test)
    train_transformed_output = train_transformed_X.with_columns(y_train)
    test_transformed_output = test_transformed_X.with_columns(y_test)
    save_(to_store=X_train, path=configs.data.paths.X_train, format='parquet')
    save_(to_store=y_train, path=configs.data.paths.y_train, format='parquet')
    save_(to_store=X_test, path=configs.data.paths.X_test, format='parquet')
    save_(to_store=y_test, path=configs.data.paths.y_test, format='parquet')
    save_(to_store=train_transformed_output, path=train_preprocessed_data_path, format='parquet')
    save_(to_store=test_transformed_output, path=test_preprocessed_data_path, format='parquet')
    save_(to_store=preprocessor, path=preprocessor_path, format='model')
    print('Preprocessing Completed..')