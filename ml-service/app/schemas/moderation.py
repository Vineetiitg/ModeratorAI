"""
SafeChat — Moderation API Schemas (Pydantic v2)
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    """Request body for POST /api/v1/moderate"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to moderate")
    channel_id: Optional[str] = Field(None, description="Channel ID for policy lookup")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    context: Optional[list[str]] = Field(default_factory=list, description="List of previous messages for conversation context")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "text": "tu bahut bada bewakoof hai bro",
                "channel_id": "general",
                "user_id": "user-123",
                "context": ["Hi, how are you?", "I am fine, but you are annoying"]
            }
        ]
    }}


class ToxicityCategories(BaseModel):
    """Breakdown of toxicity scores by category."""
    toxic: float = Field(ge=0, le=1)
    severe_toxic: float = Field(ge=0, le=1)
    obscene: float = Field(ge=0, le=1)
    identity_hate: float = Field(ge=0, le=1)
    insult: float = Field(ge=0, le=1)
    threat: float = Field(ge=0, le=1)


class ModerationResponse(BaseModel):
    """Response body for POST /api/v1/moderate"""
    is_toxic: bool
    overall_score: float = Field(ge=0, le=1)
    severity: str = Field(description="SAFE | LOW | MEDIUM | HIGH")
    categories: Dict[str, float]
    detected_language: str
    suggestion: Optional[str] = Field(None, description="Polite alternative (if toxic)")
    model_version: str
    inference_time_ms: int


class DetoxifyRequest(BaseModel):
    """Request body for POST /api/v1/detoxify"""
    text: str = Field(..., min_length=1, max_length=5000)
    target_language: Optional[str] = Field(None, description="Override language detection")
    context: Optional[list[str]] = Field(default_factory=list, description="Conversation context for intent preservation")


class DetoxifyResponse(BaseModel):
    """Response body for POST /api/v1/detoxify"""
    original: str
    detoxified: str
    method: str = Field(description="'gemini' or 'fallback'")
    language: str
    confidence: float = Field(ge=0, le=1)


class BatchModerationRequest(BaseModel):
    """Request body for POST /api/v1/moderate/batch"""
    texts: list[str] = Field(..., min_length=1, max_length=50)
    channel_id: Optional[str] = None
    user_id: Optional[str] = None


class BatchModerationResponse(BaseModel):
    """Response body for POST /api/v1/moderate/batch"""
    results: list[ModerationResponse]
    total_inference_time_ms: int
