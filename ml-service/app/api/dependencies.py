"""
SafeChat — FastAPI Dependencies

Shared dependencies injected into API route handlers.
"""

from fastapi import Depends, HTTPException

from app.models.model_manager import model_manager


async def require_models_ready():
    """
    Dependency that ensures ML models are loaded before processing requests.
    Raises 503 if models aren't ready yet.
    """
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail="ML models are still loading. Please try again in a moment.",
        )
    return model_manager
