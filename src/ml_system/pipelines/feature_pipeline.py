from ml_system.scripts.data.loader import fetch_raw_data

class FeaturePipeline:
    def __init__(self, configs):
        self.configs = configs
    def execute(self):
        fetch_raw_data()
        print("Feature Pipeline Done..")
        