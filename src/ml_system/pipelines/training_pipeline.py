from ml_system.scripts.features.feature_store import fetch_from_hops
from ml_system.scripts.models.xgb import train_model

class TrainingPipeline:
    def __init__(self, configs):
        self.configs = configs
    def execute(self):
        # fetch_from_hops(configs=self.configs)
        train_model(configs=self.configs)
        print("Training Pipeline Done..")
        