from ml_system.pipelines.inference_pipeline import InferencePipeline
from fastapi import FastAPI, HTTPException
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pydantic import BaseModel, Field

app = FastAPI()

# Global variables
config = None
inference_pipeline = None

class PredictionRequest(BaseModel):
    gender: str = Field(..., example="Male")
    platform: str = Field(..., example="Instagram")
    age: int = Field(..., example=25, gt=0)
    daily_screen_time_min: float = Field(..., example=180.0, ge=0)
    social_media_time_min: float = Field(..., example=120.0, ge=0)
    negative_interactions_count: int = Field(..., example=5, ge=0)
    positive_interactions_count: int = Field(..., example=15, ge=0)
    sleep_hours: float = Field(..., example=7.5, ge=0, le=24)
    physical_activity_min: float = Field(..., example=30.0, ge=0)
    anxiety_level: int = Field(..., example=3, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "platform": "Instagram",
                "age": 28,
                "daily_screen_time_min": 240.0,
                "social_media_time_min": 180.0,
                "negative_interactions_count": 8,
                "positive_interactions_count": 12,
                "sleep_hours": 6.5,
                "physical_activity_min": 20.0,
                "anxiety_level": 4
            }
        }

@app.on_event("startup")
async def startup_event():
    global config, inference_pipeline
    
    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent.parent / "configs")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    config = compose(config_name="config")
    inference_pipeline = InferencePipeline(config)
    print("✅ Inference pipeline initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    GlobalHydra.instance().clear()

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized")
    
    try:
        # Convert Pydantic model to dictionary
        features = request.model_dump()  # This converts PredictionRequest to dict
        result = inference_pipeline.execute(input=features)
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")