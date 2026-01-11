from ml_system.scripts.features.feature_store import get_registered_model
from omegaconf import DictConfig
import polars as pl
import joblib
import shutil
from typing import Any

def infer(configs: DictConfig, input: Any):
    model = get_registered_model(configs=configs)
    shutil.rmtree(configs.models.paths.registry_model)
    model.download(configs.models.paths.registry_model)
    
    # Load the model
    loaded_model = joblib.load(configs.models.paths.registry_model + '/full_model_pipeline.joblib')
    
    # Convert input dictionary to DataFrame
    # Wrap each value in a list to create a single-row DataFrame
    input_data_dict = {key: [value] for key, value in input.items()}
    input_data = pl.DataFrame(input_data_dict)
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    return f"{prediction[0]}"