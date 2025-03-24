"""Prediction endpoints for the supply chain forecaster API."""

from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from api.models.model_service import ModelService
from utils import ApplicationError, ModelError, get_logger

logger = get_logger(__name__)
router = APIRouter()


class PredictionParams(BaseModel):
    """Parameters for making predictions with a trained model."""
    
    model_name: str = Field(..., description="Name of the model to use")
    model_type: str = Field(..., description="Type of the model")
    feature_columns: List[str] = Field(..., description="Feature column names")
    from_deployment: bool = Field(True, description="Whether to load model from deployment")
    prediction_params: Optional[Dict] = Field(None, description="Prediction-specific parameters")


def get_model_service():
    """
    Dependency to get the model service.
    
    Returns:
        Model service instance.
    """
    return ModelService()


async def read_csv_file(file: UploadFile) -> pd.DataFrame:
    """
    Read CSV file into a pandas DataFrame.
    
    Args:
        file: Uploaded CSV file.
    
    Returns:
        DataFrame containing the data.
    """
    try:
        contents = await file.read()
        return pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading CSV file: {str(e)}",
        )


@router.post("/", summary="Make predictions with a trained model")
async def predict(
    params: PredictionParams,
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Make predictions using a trained model.
    
    Args:
        params: Prediction parameters.
        file: CSV file containing the feature data.
    
    Returns:
        Prediction results.
    """
    try:
        # Read data
        df = await read_csv_file(file)
        
        # Validate columns
        for column in params.feature_columns:
            if column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{column}' not found in data",
                )
        
        # Prepare features
        X = df[params.feature_columns]
        
        # Load model
        model = model_service.load_model(
            params.model_name, params.model_type, params.from_deployment
        )
        
        # Make predictions
        predictions = model.predict(X, **(params.prediction_params or {}))
        
        return {
            "status": "success",
            "model_name": params.model_name,
            "predictions": predictions.tolist(),
        }
    
    except ModelError as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}",
        )
