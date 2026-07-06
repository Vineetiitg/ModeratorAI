"""
SafeChat API Gateway — Authentication & User Schemas
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class RegisterRequest(BaseModel):
    """Payload for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")
    display_name: str = Field(..., min_length=2, max_length=50, description="Display name")


class LoginRequest(BaseModel):
    """Payload for user login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response returning JWT access token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """Public user profile DTO."""
    id: str
    email: str
    display_name: str
    role: str
    active: bool
    is_muted: bool
    violation_count: int
    created_at: str

    class Config:
        from_attributes = True
