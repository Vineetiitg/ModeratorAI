"""
SafeChat API Gateway — Authentication Service (PostgreSQL)

Handles user registration, login verification, password hashing, and JWT creation
using asynchronous SQLAlchemy queries against PostgreSQL.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from safety_platform.models.postgres import User
from safety_platform.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from safety_platform.core.security import get_password_hash, verify_password, create_access_token
from safety_platform.core.config import settings
from loguru import logger


class AuthService:
    """Service handling PostgreSQL user authentication and JWT token generation."""

    @staticmethod
    async def register_user(db: AsyncSession, payload: RegisterRequest) -> User:
        """Register a new user in PostgreSQL after checking email uniqueness."""
        stmt = select(User).where(User.email == payload.email.lower())
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists."
            )

        hashed = get_password_hash(payload.password)
        new_user = User(
            email=payload.email.lower(),
            password_hash=hashed,
            display_name=payload.display_name,
            role="USER",
            active=True,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        logger.info(f"Registered new PostgreSQL user: {new_user.email} (ID: {new_user.id})")
        return new_user

    @staticmethod
    async def login_user(db: AsyncSession, payload: LoginRequest) -> TokenResponse:
        """Authenticate user against PostgreSQL and return a JWT access token."""
        stmt = select(User).where(User.email == payload.email.lower())
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated."
            )

        access_token = create_access_token(subject=user.id)
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        """Fetch a user by ID from PostgreSQL."""
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
