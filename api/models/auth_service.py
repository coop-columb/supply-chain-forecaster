"""Authentication service for the supply chain forecaster API."""

import hashlib
import hmac
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials

from utils import get_logger

logger = get_logger(__name__)

# Security scheme dependencies
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
basic_auth = HTTPBasic(auto_error=False)


class AuthService:
    """Service for API authentication and authorization."""
    
    def __init__(
        self,
        secret_key: str = None,
        api_keys_file: str = None,
        token_expire_minutes: int = 60,
    ):
        """
        Initialize the authentication service.
        
        Args:
            secret_key: Secret key for JWT signing. Default is from environment or generated.
            api_keys_file: Path to file with API keys. Default is from environment.
            token_expire_minutes: Expiration time for JWT tokens in minutes.
        """
        # Get or generate secret key
        self.secret_key = secret_key or os.environ.get(
            "SECRET_KEY", secrets.token_hex(32)
        )
        
        # Get API keys file path
        self.api_keys_file = api_keys_file or os.environ.get(
            "API_KEYS_FILE", "api_keys.json"
        )
        
        # Token expiration
        self.token_expire_minutes = token_expire_minutes
        
        # Load API keys
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """
        Load API keys from file.
        
        Returns:
            Dictionary mapping API keys to key metadata.
        """
        import json
        
        try:
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"API keys file not found: {self.api_keys_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
            return {}
    
    def _save_api_keys(self) -> bool:
        """
        Save API keys to file.
        
        Returns:
            True if successful, False otherwise.
        """
        import json
        
        try:
            with open(self.api_keys_file, "w") as f:
                json.dump(self.api_keys, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")
            return False
    
    def authenticate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key to authenticate.
        
        Returns:
            API key metadata if valid, None otherwise.
        """
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        
        # Check expiration
        if key_data.get("expires_at") and time.time() > key_data["expires_at"]:
            logger.warning(f"Expired API key used: {api_key[:8]}...")
            return None
        
        # Update last used timestamp
        key_data["last_used"] = int(time.time())
        
        return key_data
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            plain_password: Plain text password.
            hashed_password: Stored password hash.
        
        Returns:
            True if password matches, False otherwise.
        """
        import bcrypt
        
        try:
            # Handle both string and bytes
            if isinstance(hashed_password, str):
                hashed_password = hashed_password.encode()
            
            return bcrypt.checkpw(
                plain_password.encode(), hashed_password
            )
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False
    
    def hash_password(self, password: str) -> str:
        """
        Hash password for storage.
        
        Args:
            password: Plain text password.
        
        Returns:
            Hashed password.
        """
        import bcrypt
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    
    def authenticate_basic(
        self, credentials: HTTPBasicCredentials
    ) -> Optional[Dict]:
        """
        Authenticate using HTTP Basic Auth.
        
        Args:
            credentials: HTTP Basic credentials.
        
        Returns:
            User data if authenticated, None otherwise.
        """
        if not credentials:
            return None
        
        # In a real system, we would look up user credentials in a database
        # For now, we'll use a simple dictionary with hardcoded users
        # This should be replaced with a proper user database
        users = {
            "admin": {
                "username": "admin",
                "full_name": "Administrator",
                "email": "admin@example.com",
                "hashed_password": self.hash_password("adminpassword"),
                "disabled": False,
                "roles": ["admin"],
            },
            "user": {
                "username": "user",
                "full_name": "Regular User",
                "email": "user@example.com",
                "hashed_password": self.hash_password("userpassword"),
                "disabled": False,
                "roles": ["user"],
            },
        }
        
        # Get user data
        user = users.get(credentials.username)
        if not user:
            return None
        
        # Check if user is disabled
        if user.get("disabled", False):
            return None
        
        # Verify password
        if not self.verify_password(credentials.password, user["hashed_password"]):
            return None
        
        return user
    
    def create_access_token(self, data: Dict) -> str:
        """
        Create JWT access token.
        
        Args:
            data: Data to encode in token.
        
        Returns:
            JWT token.
        """
        to_encode = data.copy()
        
        # Set expiration
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        to_encode.update({"exp": expire})
        
        # Encode token
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token.
        
        Returns:
            Token data if valid, None otherwise.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError as e:
            logger.warning(f"JWT validation error: {str(e)}")
            return None
    
    def create_api_key(
        self, name: str, expires_days: Optional[int] = None, scope: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Create a new API key.
        
        Args:
            name: Name for this API key.
            expires_days: Number of days until key expires (None for no expiration).
            scope: Permission scope for this key.
        
        Returns:
            Tuple of (API key, metadata).
        """
        # Generate a new key
        api_key = secrets.token_hex(32)
        
        # Create metadata
        key_data = {
            "name": name,
            "created_at": int(time.time()),
            "last_used": None,
        }
        
        # Add expiration if specified
        if expires_days is not None:
            key_data["expires_at"] = int(
                time.time() + (expires_days * 24 * 60 * 60)
            )
        
        # Add scope if specified
        if scope:
            key_data["scope"] = scope
        
        # Save to memory
        self.api_keys[api_key] = key_data
        
        # Save to file
        self._save_api_keys()
        
        return api_key, key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke.
        
        Returns:
            True if key was revoked, False otherwise.
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            self._save_api_keys()
            return True
        return False
    
    def rate_limit_check(
        self, request: Request, key_data: Dict
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check rate limits for API key.
        
        Args:
            request: Request object.
            key_data: API key metadata.
        
        Returns:
            Tuple of (allowed, limit_info).
        """
        # Rate limiting could be implemented here based on key_data
        # For now, we'll return a placeholder implementation
        return True, {
            "limit": 1000,
            "remaining": 999,
            "reset": int(time.time()) + 3600,
        }


# Create a global auth service instance
auth_service = AuthService()


def get_auth_service() -> AuthService:
    """
    Dependency to get the auth service.
    
    Returns:
        Auth service instance.
    """
    return auth_service


async def get_current_user(
    api_key: str = Depends(api_key_header),
    credentials: HTTPBasicCredentials = Depends(basic_auth),
    auth_service: AuthService = Depends(get_auth_service),
) -> Dict:
    """
    Get current authenticated user using either API key or Basic Auth.
    
    Args:
        api_key: API key from header.
        credentials: HTTP Basic credentials.
        auth_service: Auth service instance.
    
    Returns:
        User data.
    
    Raises:
        HTTPException: If authentication fails.
    """
    # Try API key first
    if api_key:
        key_data = auth_service.authenticate_api_key(api_key)
        if key_data:
            # For API keys, we'll create a simple user object
            return {
                "username": key_data.get("name", "api-user"),
                "auth_type": "api_key",
                "key_data": key_data,
            }
    
    # Then try basic auth
    if credentials:
        user = auth_service.authenticate_basic(credentials)
        if user:
            return {
                **user,
                "auth_type": "basic",
            }
    
    # Authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Basic"},
    )


async def get_optional_user(
    api_key: str = Depends(api_key_header),
    credentials: HTTPBasicCredentials = Depends(basic_auth),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[Dict]:
    """
    Get current user if authentication provided, but don't require it.
    
    Args:
        api_key: API key from header.
        credentials: HTTP Basic credentials.
        auth_service: Auth service instance.
    
    Returns:
        User data if authenticated, None otherwise.
    """
    try:
        return await get_current_user(api_key, credentials, auth_service)
    except HTTPException:
        return None