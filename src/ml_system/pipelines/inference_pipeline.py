from ml_system.scripts.inference.make_predictions import infer
from omegaconf import OmegaConf

class InferencePipeline:
    def __init__(self, configs):
        self.configs = configs
    def execute(self, input):
        predictions = infer(configs=self.configs, input=input)
        print("Inference Pipeline Done..")
        
if __name__ == "__main__":
    configs = OmegaConf.create({"models": {"registry": {"name": "xgb_model", "version": 1}, "paths": {"registry_model": "artifacts/model/registry_model"}}})
    inst = InferencePipeline(configs=configs)
    inst.execute({"a": 2})