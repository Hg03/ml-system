from ml_system.scripts.data.loader import from_supabase
from ml_system.scripts.validation.data_validator import raw_validation

class FeaturePipeline:
    def __init__(self, configs):
        self.configs = configs

    def execute(self):
        from_supabase(configs=self.configs)
        raw_validation(configs=self.configs)
        print("Feature Pipeline Done..")
