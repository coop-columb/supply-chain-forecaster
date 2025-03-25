"""Anomaly detection endpoints for the supply chain forecaster API."""

from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from api.models.anomaly_service import AnomalyService
from utils import ApplicationError, ModelError, get_logger

logger = get_logger(__name__)
router = APIRouter()


class AnomalyTrainingParams(BaseModel):
    """Parameters for training an anomaly detection model."""

    model_type: str = Field(..., description="Type of model to train")
    model_name: Optional[str] = Field(None, description="Name for the model")
    feature_columns: List[str] = Field(..., description="Feature column names")
    target_column: Optional[str] = Field(
        None, description="Target column for supervised models"
    )
    model_params: Optional[Dict] = Field(None, description="Model-specific parameters")
    training_params: Optional[Dict] = Field(None, description="Training parameters")
    save_model: bool = Field(True, description="Whether to save the trained model")


class AnomalyDetectionParams(BaseModel):
    """Parameters for detecting anomalies."""

    model_name: str = Field(..., description="Name of the model to use")
    model_type: str = Field(..., description="Type of the model")
    feature_columns: List[str] = Field(..., description="Feature column names")
    threshold: Optional[float] = Field(
        None, description="Threshold for anomaly detection"
    )
    from_deployment: bool = Field(
        True, description="Whether to load model from deployment"
    )
    detection_params: Optional[Dict] = Field(
        None, description="Detection-specific parameters"
    )
    return_details: bool = Field(
        True, description="Whether to return detailed anomaly information"
    )


def get_anomaly_service():
    """
    Dependency to get the anomaly detection service.

    Returns:
        Anomaly detection service instance.
    """
    return AnomalyService()


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


@router.post("/train", summary="Train an anomaly detection model")
async def train_model(
    params: AnomalyTrainingParams,
    file: UploadFile = File(...),
    anomaly_service: AnomalyService = Depends(get_anomaly_service),
):
    """
    Train an anomaly detection model on the provided data.

    Args:
        params: Training parameters.
        file: CSV file containing the data.

    Returns:
        Training results.
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

        # Check target column if provided
        y = None
        if params.target_column:
            if params.target_column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Target column '{params.target_column}' not found in data",
                )
            y = df[params.target_column]

        # Prepare features
        X = df[params.feature_columns]

        # Train model
        model, metrics = anomaly_service.train_model(
            params.model_type,
            X,
            y,
            model_params=params.model_params,
            model_name=params.model_name,
            save_model=params.save_model,
            **(params.training_params or {}),
        )

        return {
            "status": "success",
            "model_name": model.name,
            "model_type": params.model_type,
            "metrics": metrics,
            "message": f"Anomaly detection model '{model.name}' trained successfully",
        }

    except ModelError as e:
        logger.error(f"Error training anomaly model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error training anomaly model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training anomaly model: {str(e)}",
        )


@router.post("/detect", summary="Detect anomalies in data")
async def detect_anomalies(
    params: AnomalyDetectionParams,
    file: UploadFile = File(...),
    anomaly_service: AnomalyService = Depends(get_anomaly_service),
):
    """
    Detect anomalies in the provided data.

    Args:
        params: Anomaly detection parameters.
        file: CSV file containing the data.

    Returns:
        Anomaly detection results.
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

        # Detect anomalies
        results = anomaly_service.detect_anomalies(
            params.model_name,
            params.model_type,
            X,
            threshold=params.threshold,
            from_deployment=params.from_deployment,
            return_details=params.return_details,
            **(params.detection_params or {}),
        )

        return {
            "status": "success",
            "model_name": params.model_name,
            "results": results,
        }

    except ModelError as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting anomalies: {str(e)}",
        )
