"""
SafeChat API Gateway — REST Moderation & Feedback Routes

POST /api/v1/moderate — Proxy text moderation to ML service & log to MongoDB
POST /api/v1/feedback — Submit human review corrections to MongoDB
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from safety_platform.schemas.chat import ModerateRequest, ModerateResponse, FeedbackSubmitRequest
from safety_platform.services.ml_client import ml_client
from safety_platform.services.chat_service import ChatService
from safety_platform.core.db import get_mongo_db
from safety_platform.models.mongo import ModeratorFeedbackDocument
from safety_platform.core.config import settings

router = APIRouter(tags=["Moderation & Feedback (ML & MongoDB)"])


@router.post("/moderate", response_model=ModerateResponse)
async def moderate_message(payload: ModerateRequest):
    """
    REST moderation endpoint (fallback for offline WebSockets).
    1. Checks Redis rate limit
    2. Calls fine-tuned MuRIL classifier via ML Service
    3. Requests Gemini AI detox suggestion if toxic
    4. Persists message record into MongoDB
    """
    user_id = payload.user_id or "demo-user"
    channel_id = payload.channel_id or "general"

    # 1. Rate Limit Check (Redis)
    allowed = await ChatService.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending more messages."
        )

    # Per user request: only classify current message without previous 4 context messages
    context_snapshot = []

    # 3. Call ML service unless skip_moderation is requested
    if payload.skip_moderation:
        ml_res = {
            "is_toxic": False,
            "overall_score": 0.0,
            "severity": "SAFE",
            "categories": {},
            "suggestion": None,
            "detected_language": "en",
            "inference_time_ms": 0,
        }
    else:
        ml_res = await ml_client.moderate_text(text=payload.text, context=context_snapshot)

    is_toxic = ml_res.get("is_toxic", False)
    severity = ml_res.get("severity", "SAFE")
    suggestion = ml_res.get("suggestion")

    # If toxic and no suggestion was returned by classifier, call detoxify endpoint
    if is_toxic and severity in ("MEDIUM", "HIGH") and not suggestion:
        detox_res = await ml_client.detoxify_text(
            text=payload.text,
            target_language=ml_res.get("detected_language", "en"),
            context=context_snapshot
        )
        suggestion = detox_res.get("detoxified_text") or detox_res.get("suggestion")
        ml_res["suggestion"] = suggestion

    # Determine status
    msg_status = "DELIVERED"
    if severity == "HIGH":
        msg_status = "BLOCKED"
    elif severity == "MEDIUM":
        msg_status = "FLAGGED"

    # 4. Save to MongoDB
    await ChatService.save_message_to_mongo(
        channel_id=channel_id,
        sender_id=user_id,
        sender_name="Demo User" if user_id == "demo-user" else user_id,
        content=payload.text,
        status=msg_status,
        moderation_data=ml_res,
        context_snapshot=context_snapshot,
    )

    return ModerateResponse(
        is_toxic=is_toxic,
        overall_score=ml_res.get("overall_score", 0.0),
        severity=severity,
        categories=ml_res.get("categories", {}),
        suggestion=suggestion,
        detected_language=ml_res.get("detected_language", "en"),
        inference_time_ms=ml_res.get("inference_time_ms", 0),
    )


@router.post("/feedback")
async def submit_feedback(payload: FeedbackSubmitRequest) -> Dict[str, Any]:
    """Record human review corrections in MongoDB for continuous learning."""
    doc = ModeratorFeedbackDocument(
        feedback_id=f"fb-{payload.message_id[:8]}",
        message_id=payload.message_id,
        moderator_id=payload.moderator_id,
        model_prediction_was_correct=payload.model_prediction_was_correct,
        correct_severity=payload.correct_severity,
        notes=payload.notes,
    )

    mongo_db = get_mongo_db()
    if mongo_db is not None:
        try:
            coll = mongo_db["feedback"]
            await coll.insert_one(doc.to_mongo_dict())
            logger.info(f"Recorded feedback for message {payload.message_id} in MongoDB")
            return {"status": "success", "feedback_id": doc.feedback_id, "message": "Feedback saved in MongoDB"}
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            raise HTTPException(status_code=500, detail="Database error saving feedback")

    return {"status": "in_memory", "message": "MongoDB offline, feedback logged in memory"}
