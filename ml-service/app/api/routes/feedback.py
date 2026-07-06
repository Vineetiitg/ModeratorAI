"""
SafeChat — Feedback API Route

POST /api/v1/feedback       — Submit moderator feedback
GET  /api/v1/feedback/stats — Get feedback statistics
"""

from fastapi import APIRouter, HTTPException

from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats
from app.services.feedback_service import feedback_service

router = APIRouter(prefix="/api/v1", tags=["Feedback & Continuous Learning"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit moderator feedback on a moderation decision.

    This feedback is used for:
      1. Tracking model accuracy over time
      2. Collecting training data for model retraining
      3. Triggering automatic retraining when threshold is reached
    """
    try:
        result = await feedback_service.submit_feedback(request.model_dump())
        return FeedbackResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """
    Get feedback statistics including model accuracy and retraining progress.

    Shows:
      - Total feedback count
      - Model accuracy (correct / total)
      - Progress toward next retraining trigger
    """
    try:
        stats = await feedback_service.get_stats()
        return FeedbackStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
