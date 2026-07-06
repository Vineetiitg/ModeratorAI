"""
SafeChat API Gateway — Chat & Moderation Schemas
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModerateRequest(BaseModel):
    """Payload sent by client when moderating text via REST API."""
    text: str = Field(..., description="Message text to moderate")
    channel_id: Optional[str] = Field("general", description="Target channel ID")
    user_id: Optional[str] = Field("demo-user", description="Sender user ID")
    skip_moderation: Optional[bool] = Field(False, description="Skip moderation check if message is a pre-approved rewrite")


class ModerateResponse(BaseModel):
    """Response returned by REST API moderation endpoint."""
    is_toxic: bool
    overall_score: float
    severity: str  # SAFE | LOW | MEDIUM | HIGH
    categories: Dict[str, float] = Field(default_factory=dict)
    suggestion: Optional[str] = None
    detected_language: Optional[str] = "en"
    inference_time_ms: int = 0


class WebSocketMessagePayload(BaseModel):
    """Incoming WebSocket JSON message from client."""
    type: str = "message"
    text: str
    channel_id: Optional[str] = "general"
    user_id: Optional[str] = None


class FeedbackSubmitRequest(BaseModel):
    """Payload for submitting moderator correction feedback."""
    message_id: str
    moderator_id: str = "mod-1"
    model_prediction_was_correct: bool = True
    correct_severity: str = "SAFE"
    notes: Optional[str] = None
