"""Model endpoints for the supply chain forecaster API."""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.models.model_service import ModelService
from utils import ApplicationError, ModelError, get_logger

logger = get_logger(__name__)
router = APIRouter()


def get_model_service():
    """
    Dependency to get the model service.
    
    Returns:
        Model service instance.
    """
    return ModelService()


@router.get("/", summary="List available models")
async def list_models(
    trained: bool = Query(False, description="List trained models"),
    deployed: bool = Query(False, description="List deployed models"),
    model_service: ModelService = Depends(get_model_service),
):
    """
    List available models.
    
    Returns:
        List of available model types or trained/deployed models.
    """
    try:
        if trained:
            return {"trained_models": model_service.get_trained_models()}
        elif deployed:
            return {"deployed_models": model_service.get_deployed_models()}
        else:
            return {"available_models": model_service.get_available_models()}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}",
        )


@router.get("/{model_name}", summary="Get model details")
async def get_model(
    model_name: str,
    model_type: str = Query(..., description="Type of model to load"),
    from_deployment: bool = Query(
        True, description="Whether to load from deployment directory"
    ),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Get details for a specific model.
    
    Args:
        model_name: Name of the model.
        model_type: Type of the model.
        from_deployment: Whether to load from deployment directory.
    
    Returns:
        Model details.
    """
    try:
        model = model_service.load_model(model_name, model_type, from_deployment)
        return {"name": model.name, "type": model_type, "metadata": model.metadata}
    except ModelError as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}",
        )


@router.post("/{model_name}/deploy", summary="Deploy a model")
async def deploy_model(
    model_name: str,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Deploy a model from training to deployment.
    
    Args:
        model_name: Name of the model to deploy.
    
    Returns:
        Deployment status.
    """
    try:
        metadata = model_service.deploy_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_name}' deployed successfully",
            "metadata": metadata,
        }
    except ModelError as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deploying model: {str(e)}",
        )


@router.delete("/{model_name}", summary="Delete a model")
async def delete_model(
    model_name: str,
    from_deployment: bool = Query(
        False, description="Whether to delete from deployment directory"
    ),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Delete a model.
    
    Args:
        model_name: Name of the model to delete.
        from_deployment: Whether to delete from deployment directory.
    
    Returns:
        Deletion status.
    """
    try:
        model_service.delete_model(model_name, from_deployment)
        return {
            "status": "success",
            "message": f"Model '{model_name}' deleted successfully"
        }
    except ModelError as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}",
        )
