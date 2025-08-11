"""
Authentication system for the AI Testing Agent.
"""

import os
import jwt
import time
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import wraps
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .secrets_manager import get_secrets_manager, SecretNames

logger = logging.getLogger(__name__)
security = HTTPBearer()

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.secrets_mgr = get_secrets_manager()
        self.jwt_secret = self._get_jwt_secret()
        self.token_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS', '24'))
    
    def _get_jwt_secret(self) -> str:
        """Get JWT secret from secrets manager or environment"""
        try:
            secret_data = self.secrets_mgr.get_secret(SecretNames.JWT_SECRET)
            if secret_data and 'secret' in secret_data:
                return secret_data['secret']
        except Exception as e:
            logger.warning(f"Failed to get JWT secret from secrets manager: {e}")
        
        # Fallback to environment variable
        return os.getenv('JWT_SECRET', 'default-secret-change-in-production')
    
    def create_access_token(self, user_id: str, username: str, roles: list = None) -> str:
        """Create a JWT access token"""
        if roles is None:
            roles = ['user']
        
        payload = {
            'user_id': user_id,
            'username': username,
            'roles': roles,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return self.hash_password(password) == hashed

# Global auth manager instance
auth_manager = AuthManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload

def require_role(required_role: str):
    """Decorator to require a specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: Dict = Depends(get_current_user), **kwargs):
            user_roles = current_user.get('roles', [])
            if required_role not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{required_role}' required"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

def require_admin(func):
    """Decorator to require admin role"""
    return require_role('admin')(func)

def require_tester(func):
    """Decorator to require tester role"""
    return require_role('tester')(func)

# Example user management (in production, use a proper database)
class UserManager:
    """Simple user management for demo purposes"""
    
    def __init__(self):
        self.secrets_mgr = get_secrets_manager()
        self.users = self._load_users()
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from secrets manager or create defaults"""
        try:
            users_data = self.secrets_mgr.get_secret('users/default')
            if users_data:
                return users_data
        except Exception as e:
            logger.warning(f"Failed to load users from secrets manager: {e}")
        
        # Default users for development
        default_users = {
            'admin': {
                'username': 'admin',
                'password_hash': auth_manager.hash_password('admin123'),
                'roles': ['admin', 'tester', 'user'],
                'email': 'admin@example.com'
            },
            'tester': {
                'username': 'tester',
                'password_hash': auth_manager.hash_password('tester123'),
                'roles': ['tester', 'user'],
                'email': 'tester@example.com'
            },
            'user': {
                'username': 'user',
                'password_hash': auth_manager.hash_password('user123'),
                'roles': ['user'],
                'email': 'user@example.com'
            }
        }
        
        # Save default users
        try:
            self.secrets_mgr.set_secret('users/default', default_users)
        except Exception as e:
            logger.warning(f"Failed to save default users: {e}")
        
        return default_users
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        user = self.users.get(username)
        if user and auth_manager.verify_password(password, user['password_hash']):
            return {
                'user_id': username,
                'username': user['username'],
                'roles': user['roles'],
                'email': user['email']
            }
        return None
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        return self.users.get(username)
    
    def create_user(self, username: str, password: str, email: str, roles: list = None) -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        if roles is None:
            roles = ['user']
        
        new_user = {
            'username': username,
            'password_hash': auth_manager.hash_password(password),
            'roles': roles,
            'email': email
        }
        
        self.users[username] = new_user
        
        try:
            self.secrets_mgr.set_secret('users/default', self.users)
            return True
        except Exception as e:
            logger.error(f"Failed to save user: {e}")
            return False

# Global user manager instance
user_manager = UserManager()

# Authentication endpoints
def login_user(username: str, password: str) -> Optional[str]:
    """Login a user and return JWT token"""
    user = user_manager.authenticate_user(username, password)
    if user:
        return auth_manager.create_access_token(
            user_id=user['user_id'],
            username=user['username'],
            roles=user['roles']
        )
    return None

# Example usage
if __name__ == "__main__":
    # Test authentication
    print("Testing authentication system...")
    
    # Test login
    token = login_user('admin', 'admin123')
    if token:
        print(f"Login successful. Token: {token[:50]}...")
        
        # Test token verification
        payload = auth_manager.verify_token(token)
        print(f"Token payload: {payload}")
    else:
        print("Login failed") 