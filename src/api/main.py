from fastapi import FastAPI
from src.api.pydantic_models import RiskRequest, RiskResponse
from joblib import load
from src.utils.preprocessing_bundle import PreprocessingBundle  # this is key
import mlflow.pyfunc
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
app = FastAPI(title="Risk Prediction API")

# Load preprocessing bundle
preprocessor: PreprocessingBundle = load("models/transformers/preprocessing_bundle.pkl")

# Load trained model from MLflow Registry
# model_name = "best_risk_model"
# model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models/mlruns/747112809802961086/models/m-3d7354a6e92e4cec9225e6059a871d4d/artifacts")

@app.post("/predict", response_model=RiskResponse)
async def predict(request: RiskRequest):
    start = time.time()
    # Format incoming features as DataFrame
    raw_df = pd.DataFrame([request.features])

    # Transform the raw input into model-ready format
    model_input = preprocessor.transform(raw_df)

    # Predict probability
    prediction = model.predict(model_input)
    probability = float(prediction[0])
    
    duration = round(time.time() - start, 3)
    print(f"Request completed in {duration}s")

    return RiskResponse(probability=probability)