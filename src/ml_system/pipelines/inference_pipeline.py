from ml_system.scripts.inference.make_predictions import infer
from omegaconf import OmegaConf

class InferencePipeline:
    def __init__(self, configs):
        self.configs = configs
    def execute(self, input):
        predictions = infer(configs=self.configs, input=input)
        print("Inference Pipeline Done..")