from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json
import os
import time
from botocore.exceptions import ClientError


app = FastAPI(title="SageMaker Shopping Prediction API")

# Get endpoint name from environment variable
ENDPOINT_NAME = 'shoppers-xgboost-endpoint'
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: str


@app.get("/health")
def health():
    """Liveness check - is the process alive and responsive?"""
    return {"status": "healthy", "version": "v3.0-cicd"}

@app.get("/ready")
def ready():
    """Readiness check - can this pod serve traffic? Verifies runtime client is initialized."""
    try:
        if runtime is not None and ENDPOINT_NAME:
            return {"status": "ready", "endpoint": ENDPOINT_NAME}
        raise HTTPException(status_code=503, detail={"status": "not ready"})
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})

def call_endpoint_with_retry(features, max_retries=3):
    """Call SageMaker endpoint with retry logic for cold starts."""
    csv_data = ",".join(str(f) for f in features)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='text/csv',
                Body=csv_data
            )
            elapsed = time.time() - start_time
            
            if elapsed > 5.0 and attempt < max_retries - 1:
                time.sleep(1)
                continue
            
            return response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ModelError' and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(
                status_code=503,
                detail=f"Endpoint error: {error_code}"
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error: {str(e)}"
                )
            time.sleep(2 ** attempt)
    
    raise HTTPException(status_code=503, detail="Endpoint unavailable after retries")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Validate feature count (adjust to match your model)
    if len(request.features) != 17:  # Adjust to your model's feature count
        raise HTTPException(
            status_code=400,
            detail=f"Expected 17 features, got {len(request.features)}"
        )
            
    # Call endpoint with retry logic
    response = call_endpoint_with_retry(request.features)
    
    # Parse response
    raw_response = response['Body'].read().decode().strip()
    prediction_value = float(raw_response)


    # Determine confidence level
    confidence = "high" if prediction_value > 0.7 else "medium" if prediction_value > 0.3 else "low"
            
    return PredictionResponse(
        prediction=prediction_value,
        confidence=confidence
    )