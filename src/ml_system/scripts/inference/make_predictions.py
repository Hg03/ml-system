from ml_system.scripts.features.feature_store import get_registered_model
from omegaconf import DictConfig
from typing import Any

def infer(configs: DictConfig, input: Any):
    model = get_registered_model(configs=configs)
    model.download(configs.models.paths.registry_model)
    
    # Will perform further inference process