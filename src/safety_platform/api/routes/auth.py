"""
SafeChat API Gateway — Auth Routes (PostgreSQL)

POST /api/v1/auth/register — Register a new account in PostgreSQL
POST /api/v1/auth/login — Authenticate and receive JWT access token
GET /api/v1/auth/me — Retrieve current user profile
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from safety_platform.core.db import get_db
from safety_platform.core.security import get_current_user_required
from safety_platform.models.postgres import User
from safety_platform.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from safety_platform.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication (PostgreSQL)"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account in PostgreSQL."""
    user = await AuthService.register_user(db, payload)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user against PostgreSQL and return a JWT access token."""
    token = await AuthService.login_user(db, payload)
    return token


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user_required)):
    """Return profile of the currently authenticated user."""
    return current_user
