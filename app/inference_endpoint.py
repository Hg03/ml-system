from ml_system.pipelines.inference_pipeline import InferencePipeline
from fastapi import FastAPI

app = FastAPI()

# Global variable to store config and pipeline
config = None
inference_pipeline = None

@app.on_event("startup")
async def startup_event():
    global config, inference_pipeline
    # Load config directly without decorator
    config_path = Path(__file__).parent.parent / "configs"
    config = OmegaConf.load(config_path / "config.yaml")
    
    # Initialize pipeline once at startup
    inference_pipeline = InferencePipeline(config)

@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(data: dict):
    result = inference_pipeline.predict(data)
    return result
