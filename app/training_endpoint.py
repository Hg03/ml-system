from ml_system.pipelines.feature_pipeline import FeaturePipeline
from ml_system.pipelines.training_pipeline import TrainingPipeline
from omegaconf import DictConfig
import hydra

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.pipeline.stage == 'feature':
        FeaturePipeline(cfg).execute()
    elif cfg.pipeline.stage == 'train':
        TrainingPipeline(cfg).execute()
    else:
        FeaturePipeline(cfg).execute()
        TrainingPipeline(cfg).execute()

if __name__ == '__main__':
    main()


