"""
SafeChat API Gateway — Security & Authentication Utilities

Handles bcrypt password hashing, JWT token generation, and OAuth2 bearer token
validation for API and WebSocket endpoints.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from fastapi import Depends, HTTPException, status, WebSocket, WebSocketException
from fastapi.security import OAuth2PasswordBearer
import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from safety_platform.core.config import settings
from safety_platform.core.db import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a bcrypt password hash."""
    return pwd_context.hash(password)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT access token for a given subject (user ID or email)."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"sub": str(subject), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[str]:
    """Decode a JWT token and return the subject string if valid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return str(payload.get("sub"))
    except jwt.PyJWTError:
        return None


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[Any]:
    """
    FastAPI dependency that returns the User model if valid bearer token is present,
    or None for unauthenticated access.
    """
    from safety_platform.models.postgres import User

    if not token:
        return None

    sub = decode_access_token(token)
    if not sub:
        return None

    stmt = select(User).where(User.id == sub)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user or not user.active:
        return None

    return user


async def get_current_user_required(
    user: Optional[Any] = Depends(get_current_user_optional),
) -> Any:
    """
    FastAPI dependency that requires an authenticated, active user.
    Raises HTTP 401 if unauthenticated.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials. Please login.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def authenticate_websocket(websocket: WebSocket, token: Optional[str]) -> Optional[Any]:
    """
    Authenticate a WebSocket connection using a token passed via query parameter.
    Returns User object or raises WebSocketException.
    """
    from safety_platform.models.postgres import User

    if not token:
        return None

    sub = decode_access_token(token)
    if not sub:
        return None

    # For websocket we get db manually since dependency injection differs in WS accept
    from safety_platform.core.db import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        stmt = select(User).where(User.id == sub)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        if user and user.active:
            return user
        return None
