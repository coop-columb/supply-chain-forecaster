"""Health check endpoints for the supply chain forecaster API."""

from fastapi import APIRouter, Response, status

router = APIRouter()


@router.get("/health", summary="Health check")
async def health_check():
    """
    Check if the API is healthy.
    
    Returns:
        Health status.
    """
    return {"status": "ok"}


@router.get("/health/readiness", summary="Readiness check")
async def readiness_check():
    """
    Check if the API is ready to serve requests.
    
    Returns:
        Readiness status.
    """
    return {"status": "ready"}


@router.get("/health/liveness", summary="Liveness check")
async def liveness_check():
    """
    Check if the API is alive.
    
    Returns:
        Liveness status.
    """
    return {"status": "alive"}


@router.get("/version", summary="API version")
async def version():
    """
    Get the API version.
    
    Returns:
        API version.
    """
    return {"version": "0.1.0"}
