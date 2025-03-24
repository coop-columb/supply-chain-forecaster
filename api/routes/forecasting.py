"""Forecasting endpoints for the supply chain forecaster API."""

from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from api.models.forecasting_service import ForecastingService
from api.models.model_service import ModelService
from utils import ApplicationError, ModelError, get_logger

logger = get_logger(__name__)
router = APIRouter()


class TrainingParams(BaseModel):
    """Parameters for training a forecasting model."""
    
    model_type: str = Field(..., description="Type of model to train")
    model_name: Optional[str] = Field(None, description="Name for the model")
    feature_columns: List[str] = Field(..., description="Feature column names")
    target_column: str = Field(..., description="Target column name")
    date_column: Optional[str] = Field(None, description="Date column name")
    model_params: Optional[Dict] = Field(None, description="Model-specific parameters")
    training_params: Optional[Dict] = Field(None, description="Training parameters")
    save_model: bool = Field(True, description="Whether to save the trained model")


class ForecastParams(BaseModel):
    """Parameters for generating a forecast."""
    
    model_name: str = Field(..., description="Name of the model to use")
    model_type: str = Field(..., description="Type of the model")
    feature_columns: List[str] = Field(..., description="Feature column names")
    date_column: Optional[str] = Field(None, description="Date column name")
    steps: int = Field(30, description="Number of steps to forecast")
    return_conf_int: bool = Field(False, description="Whether to return confidence intervals")
    from_deployment: bool = Field(True, description="Whether to load model from deployment")
    forecast_params: Optional[Dict] = Field(None, description="Forecast-specific parameters")


class CrossValidationParams(BaseModel):
    """Parameters for cross-validation."""
    
    model_type: str = Field(..., description="Type of model to cross-validate")
    feature_columns: List[str] = Field(..., description="Feature column names")
    target_column: str = Field(..., description="Target column name")
    date_column: Optional[str] = Field(None, description="Date column name")
    model_params: Optional[Dict] = Field(None, description="Model-specific parameters")
    strategy: str = Field("expanding", description="Cross-validation strategy")
    initial_window: int = Field(30, description="Initial window size")
    step_size: int = Field(7, description="Step size for moving forward")
    horizon: int = Field(7, description="Forecasting horizon")
    max_train_size: Optional[int] = Field(None, description="Maximum training size")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute")


def get_forecasting_service():
    """
    Dependency to get the forecasting service.
    
    Returns:
        Forecasting service instance.
    """
    return ForecastingService()


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


@router.post("/train", summary="Train a forecasting model")
async def train_model(
    params: TrainingParams,
    file: UploadFile = File(...),
    forecasting_service: ForecastingService = Depends(get_forecasting_service),
):
    """
    Train a forecasting model on the provided data.
    
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
        for column in params.feature_columns + [params.target_column]:
            if column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{column}' not found in data",
                )
        
        # Prepare features and target
        X = df[params.feature_columns]
        y = df[params.target_column]
        
        # Convert date column to datetime if provided
        if params.date_column and params.date_column in df.columns:
            df[params.date_column] = pd.to_datetime(df[params.date_column])
        
        # Train model
        model, metrics = forecasting_service.train_model(
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
            "message": f"Model '{model.name}' trained successfully",
        }
    
    except ModelError as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}",
        )


@router.post("/forecast", summary="Generate a forecast")
async def generate_forecast(
    params: ForecastParams,
    file: UploadFile = File(...),
    forecasting_service: ForecastingService = Depends(get_forecasting_service),
):
    """
    Generate a forecast using a trained model.
    
    Args:
        params: Forecast parameters.
        file: CSV file containing the feature data.
    
    Returns:
        Forecast results.
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
        
        # Convert date column to datetime if provided
        if params.date_column and params.date_column in df.columns:
            df[params.date_column] = pd.to_datetime(df[params.date_column])
        
        # Generate forecast
        forecast_result = forecasting_service.forecast(
            params.model_name,
            params.model_type,
            X,
            steps=params.steps,
            return_conf_int=params.return_conf_int,
            from_deployment=params.from_deployment,
            **(params.forecast_params or {}),
        )
        
        # Format the results
        if params.return_conf_int:
            forecast, lower, upper = forecast_result
            result = {
                "forecast": forecast.tolist(),
                "lower_bound": lower.tolist(),
                "upper_bound": upper.tolist(),
            }
        else:
            result = {"forecast": forecast_result.tolist()}
        
        # Add date index if provided
        if params.date_column and params.date_column in df.columns:
            dates = df[params.date_column].tolist()
            result["dates"] = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dates]
        
        return {
            "status": "success",
            "model_name": params.model_name,
            "steps": params.steps,
            "result": result,
        }
    
    except ModelError as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecast: {str(e)}",
        )


@router.post("/cross-validate", summary="Perform time series cross-validation")
async def cross_validate(
    params: CrossValidationParams,
    file: UploadFile = File(...),
    forecasting_service: ForecastingService = Depends(get_forecasting_service),
):
    """
    Perform time series cross-validation on a model.
    
    Args:
        params: Cross-validation parameters.
        file: CSV file containing the data.
    
    Returns:
        Cross-validation results.
    """
    try:
        # Read data
        df = await read_csv_file(file)
        
        # Validate columns
        for column in params.feature_columns + [params.target_column]:
            if column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{column}' not found in data",
                )
        
        # Convert date column to datetime if provided
        if params.date_column and params.date_column in df.columns:
            df[params.date_column] = pd.to_datetime(df[params.date_column])
        
        # Perform cross-validation
        cv_results = forecasting_service.cross_validate(
            params.model_type,
            df,
            params.target_column,
            date_col=params.date_column,
            model_params=params.model_params,
            strategy=params.strategy,
            initial_window=params.initial_window,
            step_size=params.step_size,
            horizon=params.horizon,
            max_train_size=params.max_train_size,
            metrics=params.metrics,
        )
        
        return {
            "status": "success",
            "model_type": params.model_type,
            "cv_results": cv_results,
        }
    
    except ModelError as e:
        logger.error(f"Error performing cross-validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error performing cross-validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing cross-validation: {str(e)}",
        )
