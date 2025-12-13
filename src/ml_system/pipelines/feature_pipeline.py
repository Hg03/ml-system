from ml_system.scripts.data.loader import from_supabase


class FeaturePipeline:
    def __init__(self, configs):
        self.configs = configs

    def execute(self):
        from_supabase(configs=self.configs)
        print("Feature Pipeline Done..")
