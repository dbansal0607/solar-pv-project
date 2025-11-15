"""
Solar PV Digital Twin - FastAPI Production Server

Endpoints:
- POST /predict: Single prediction from JSON features
- POST /predict_batch: Batch prediction from CSV file upload
- GET /: Health check and model info

Author: [Your Name]
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import io
import json
from datetime import datetime

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Solar PV Digital Twin API",
    description="Predict solar panel power output from environmental parameters",
    version="1.0.0"
)

# ============================================================================
# LOAD PRODUCTION MODEL ON STARTUP
# ============================================================================
try:
    print("Loading production model...")
    model_artifact = joblib.load('models/pipeline_prod.joblib')
    
    # Extract components from artifact
    pipeline = model_artifact['pipeline']
    features = model_artifact['features']
    model_version = model_artifact['version']
    
    print(f"✓ Model loaded successfully!")
    print(f"  Version: {model_version}")
    print(f"  Features: {features}")
    
except FileNotFoundError:
    print("❌ ERROR: Model file not found at 'models/pipeline_prod.joblib'")
    print("   Please run: python src/train_production.py")
    raise
except Exception as e:
    print(f"❌ ERROR loading model: {str(e)}")
    raise

# ============================================================================
# PYDANTIC MODELS FOR REQUEST VALIDATION
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Schema for single prediction requests.
    Uses Field(..., alias="...") to handle column names with special characters.
    """
    Solar_Irradiance_kWh_m2: float = Field(..., description="Solar irradiance in kWh/m²", example=0.75)
    Temperature_C: float = Field(..., description="Ambient temperature in Celsius", example=25.5)
    Wind_Speed_mps: float = Field(..., description="Wind speed in m/s", example=3.2)
    
    # CRITICAL: Handle the % character in column name using alias
    Relative_Humidity_pct: float = Field(
        ..., 
        alias="Relative_Humidity_%",  # Accept "Relative_Humidity_%" in JSON
        description="Relative humidity percentage", 
        example=65.0
    )
    
    Panel_Tilt_deg: float = Field(..., description="Panel tilt angle in degrees", example=30.0)
    Panel_Azimuth_deg: float = Field(..., description="Panel azimuth angle in degrees", example=180.0)
    Plane_of_Array_Irradiance: float = Field(..., description="POA irradiance", example=800.0)
    Cell_Temperature_C: float = Field(..., description="Cell temperature in Celsius", example=35.0)
    
    class Config:
        # Enable population by field name to support both alias and field name
        populate_by_name = True
        
        # Example JSON for API documentation
        json_schema_extra = {
            "example": {
                "Solar_Irradiance_kWh_m2": 0.75,
                "Temperature_C": 25.5,
                "Wind_Speed_mps": 3.2,
                "Relative_Humidity_%": 65.0,
                "Panel_Tilt_deg": 30.0,
                "Panel_Azimuth_deg": 180.0,
                "Plane_of_Array_Irradiance": 800.0,
                "Cell_Temperature_C": 35.0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for single predictions"""
    predicted_power: float = Field(..., description="Predicted power output in Watts")
    model_version: str = Field(..., description="Model version identifier")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    total_predictions: int = Field(..., description="Number of predictions made")
    sample_predictions: List[Dict] = Field(..., description="First 5 predictions as sample")
    model_version: str = Field(..., description="Model version identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    message: str = Field(..., description="Status message")

# ============================================================================
# ENDPOINT 1: HEALTH CHECK AND MODEL INFO
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint - returns model info and status.
    
    Returns:
        dict: API status and model information
    """
    return {
        "status": "online",
        "api_name": "Solar PV Digital Twin API",
        "model_version": model_version,
        "features_required": features,
        "endpoints": {
            "POST /predict": "Single prediction from JSON",
            "POST /predict_batch": "Batch prediction from CSV upload"
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# ENDPOINT 2: SINGLE PREDICTION
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict power output for a single set of environmental parameters.
    
    Args:
        request (PredictionRequest): Feature values as JSON
        
    Returns:
        PredictionResponse: Predicted power and metadata
        
    Example:
        POST /predict
        {
            "Solar_Irradiance_kWh_m2": 0.75,
            "Temperature_C": 25.5,
            "Wind_Speed_mps": 3.2,
            "Relative_Humidity_%": 65.0,
            "Panel_Tilt_deg": 30.0,
            "Panel_Azimuth_deg": 180.0,
            "Plane_of_Array_Irradiance": 800.0,
            "Cell_Temperature_C": 35.0
        }
    """
    try:
        # Convert Pydantic model to dict, using aliases
        # This ensures "Relative_Humidity_%" is correctly mapped
        data_dict = request.model_dump(by_alias=True)
        
        # Create DataFrame with feature values in correct order
        input_df = pd.DataFrame([data_dict])[features]
        
        # Make prediction using the pipeline
        prediction = pipeline.predict(input_df)[0]
        
        # Return response
        return PredictionResponse(
            predicted_power=round(float(prediction), 2),
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
        
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature: {str(e)}. Required features: {features}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# ============================================================================
# ENDPOINT 3: BATCH PREDICTION FROM CSV UPLOAD
# ============================================================================

@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with feature columns"),
    return_csv: bool = False
):
    """
    Predict power output for multiple samples from uploaded CSV file.
    
    Args:
        file (UploadFile): CSV file containing feature columns
        return_csv (bool): If True, return full CSV with predictions as download
        
    Returns:
        If return_csv=False: JSON with sample predictions and count
        If return_csv=True: CSV file download with predictions column added
        
    Example:
        POST /predict_batch
        Files: {"file": "unseen_data.csv"}
        Query params: {"return_csv": true}
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV. Please upload a .csv file."
        )
    
    try:
        # Read uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"✓ CSV uploaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Validate that all required features are present
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features in CSV: {list(missing_features)}. Required: {features}"
            )
        
        # Extract features in correct order
        X_batch = df[features]
        
        # Make batch predictions
        predictions = pipeline.predict(X_batch)
        
        # Add predictions to dataframe
        df['Predicted_Power_W'] = predictions.round(2)
        
        # Option 1: Return CSV file for download
        if return_csv:
            # Convert DataFrame to CSV in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            # Return as streaming response (file download)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        
        # Option 2: Return JSON summary with sample predictions
        else:
            # Get first 5 predictions as sample
            sample_size = min(5, len(df))
            sample_data = df.head(sample_size).to_dict('records')
            
            return BatchPredictionResponse(
                total_predictions=len(predictions),
                sample_predictions=sample_data,
                model_version=model_version,
                timestamp=datetime.now().isoformat(),
                message=f"Successfully predicted {len(predictions)} samples. Set return_csv=true to download full results."
            )
            
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("\n" + "="*70)
    print("🌞 Solar PV Digital Twin API - ONLINE")
    print("="*70)
    print(f"Model Version: {model_version}")
    print(f"Features: {len(features)}")
    print(f"Endpoints:")
    print(f"  GET  /          - Health check and model info")
    print(f"  POST /predict   - Single prediction from JSON")
    print(f"  POST /predict_batch - Batch prediction from CSV upload")
    print(f"  GET  /docs      - Interactive API documentation")
    print("="*70 + "\n")

# ============================================================================
# RUN SERVER (for testing only)
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)