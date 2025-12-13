from sklearn.impute import SimpleImputer
from ml_system.scripts.utils.saving_utils import save_
from ml_system.scripts.utils.loading_utils import load_
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
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
    preprocessed_data_path = configs.data.paths.processed_data
    preprocessor_path = configs.features.paths.preprocessor
    df = load_(path=raw_data_path, format='parquet')
    df, X, y = preprare_to_preprocess(df=df, configs=configs)
    preprocessor = make_preprocessor(configs=configs)
    transformed_X = preprocessor.fit_transform(X)
    transformed_output = transformed_X.with_columns(y)
    save_(to_store=transformed_output, path=preprocessed_data_path, format='parquet')
    save_(to_store=preprocessor, path=preprocessor_path, format='model')
    print('Preprocessing Completed..')
    