"""Authentication and API key management endpoints for the supply chain forecaster API."""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.models.auth_service import AuthService, get_auth_service, get_current_user
from utils import ApplicationError, get_logger

logger = get_logger(__name__)
router = APIRouter()


class ApiKeyCreate(BaseModel):
    """Model for API key creation."""

    name: str = Field(..., description="Name for this API key")
    expires_days: Optional[int] = Field(
        None, description="Days until key expires (null for no expiration)"
    )
    scope: Optional[str] = Field(None, description="Permission scope for this key")


class ApiKeyResponse(BaseModel):
    """Model for API key response."""

    key: str = Field(..., description="The API key (only shown once)")
    name: str = Field(..., description="Name for this API key")
    created_at: int = Field(..., description="Unix timestamp of creation time")
    expires_at: Optional[int] = Field(
        None, description="Unix timestamp of expiration time"
    )
    scope: Optional[str] = Field(None, description="Permission scope for this key")


class ApiKeyInfo(BaseModel):
    """Model for API key info (without the actual key)."""

    id: str = Field(..., description="ID of the API key (prefix of actual key)")
    name: str = Field(..., description="Name for this API key")
    created_at: int = Field(..., description="Unix timestamp of creation time")
    expires_at: Optional[int] = Field(
        None, description="Unix timestamp of expiration time"
    )
    last_used: Optional[int] = Field(None, description="Unix timestamp of last usage")
    scope: Optional[str] = Field(None, description="Permission scope for this key")


@router.post("/keys", response_model=ApiKeyResponse, summary="Create a new API key")
async def create_api_key(
    key_data: ApiKeyCreate,
    current_user: Dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Create a new API key.

    This endpoint allows users to create new API keys for programmatic access.
    The actual key is only returned once in the response, so make sure to save it.
    """
    try:
        # Check user permissions (only admins can create API keys)
        if current_user.get("auth_type") != "basic" or "admin" not in current_user.get(
            "roles", []
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create API keys",
            )

        # Create the API key
        api_key, key_data = auth_service.create_api_key(
            name=key_data.name,
            expires_days=key_data.expires_days,
            scope=key_data.scope,
        )

        logger.info(
            f"API key created: {key_data['name']} by {current_user['username']}"
        )

        # Return the key (only time it will be shown in full)
        return ApiKeyResponse(
            key=api_key,
            name=key_data["name"],
            created_at=key_data["created_at"],
            expires_at=key_data.get("expires_at"),
            scope=key_data.get("scope"),
        )

    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating API key: {str(e)}",
        )


@router.get("/keys", response_model=List[ApiKeyInfo], summary="List all API keys")
async def list_api_keys(
    current_user: Dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    List all API keys.

    This endpoint returns information about all API keys, but not the actual keys themselves.
    """
    try:
        # Check user permissions (only admins can list all API keys)
        if current_user.get("auth_type") != "basic" or "admin" not in current_user.get(
            "roles", []
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to list API keys",
            )

        # Format API key info
        key_info = []
        for api_key, data in auth_service.api_keys.items():
            key_info.append(
                ApiKeyInfo(
                    id=api_key[:8] + "...",  # Only show prefix
                    name=data["name"],
                    created_at=data["created_at"],
                    expires_at=data.get("expires_at"),
                    last_used=data.get("last_used"),
                    scope=data.get("scope"),
                )
            )

        return key_info

    except Exception as e:
        logger.error(f"Error listing API keys: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing API keys: {str(e)}",
        )


@router.delete("/keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(
    key_id: str,
    current_user: Dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Revoke an API key.

    This endpoint revokes an API key, making it no longer valid for authentication.
    """
    try:
        # Check user permissions (only admins can revoke API keys)
        if current_user.get("auth_type") != "basic" or "admin" not in current_user.get(
            "roles", []
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to revoke API keys",
            )

        # Find the API key by prefix
        target_key = None
        for api_key in auth_service.api_keys:
            if api_key.startswith(key_id):
                target_key = api_key
                break

        if not target_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key with ID '{key_id}' not found",
            )

        # Revoke the key
        if auth_service.revoke_api_key(target_key):
            logger.info(f"API key revoked: {key_id} by {current_user['username']}")
            return {"status": "success", "message": f"API key {key_id} revoked"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to revoke API key {key_id}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error revoking API key: {str(e)}",
        )


@router.get("/me", summary="Get current user information")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """
    Get information about the currently authenticated user.

    This endpoint returns information about the currently authenticated user.
    """
    try:
        # Remove sensitive information
        user_info = {k: v for k, v in current_user.items() if k != "hashed_password"}

        # Add formatted timestamps
        if current_user.get("auth_type") == "api_key" and "key_data" in current_user:
            key_data = current_user["key_data"]
            if "created_at" in key_data:
                user_info["created_at_formatted"] = datetime.fromtimestamp(
                    key_data["created_at"]
                ).strftime("%Y-%m-%d %H:%M:%S")

            if "expires_at" in key_data and key_data["expires_at"]:
                user_info["expires_at_formatted"] = datetime.fromtimestamp(
                    key_data["expires_at"]
                ).strftime("%Y-%m-%d %H:%M:%S")

        return user_info

    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user info: {str(e)}",
        )
