from ml_system.pipelines.inference_pipeline import InferencePipeline
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "API is running"}
