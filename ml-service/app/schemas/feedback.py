"""
SafeChat — Feedback API Schemas (Pydantic v2)

Used by the moderator dashboard to submit corrections,
which feed into the continuous learning pipeline.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request body for POST /api/v1/feedback"""
    message_id: str = Field(..., description="MongoDB ObjectId of the moderated message")
    moderator_id: str = Field(..., description="UUID of the moderator")
    model_prediction_was_correct: bool = Field(
        ..., description="Did the model classify correctly?"
    )
    correct_label: Optional[str] = Field(
        None,
        description="Correct toxicity label if model was wrong (e.g., 'not_toxic', 'toxic', 'insult')",
    )
    correct_severity: Optional[str] = Field(
        None, description="Correct severity if model was wrong (SAFE/LOW/MEDIUM/HIGH)"
    )
    notes: Optional[str] = Field(
        None, max_length=1000, description="Moderator notes explaining the correction"
    )


class FeedbackResponse(BaseModel):
    """Response body for POST /api/v1/feedback"""
    feedback_id: str
    message: str
    total_feedback_count: int
    retrain_threshold: int
    retrain_triggered: bool


class FeedbackStats(BaseModel):
    """Response body for GET /api/v1/feedback/stats"""
    total_feedback: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float = Field(ge=0, le=1)
    feedback_since_last_retrain: int
    retrain_threshold: int
    next_retrain_at: int  # Feedback count needed to trigger retrain
    last_retrain_at: Optional[datetime] = None
