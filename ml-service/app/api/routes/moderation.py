"""
SafeChat — Moderation API Route

POST /api/v1/moderate       — Moderate a single message
POST /api/v1/moderate/batch — Moderate multiple messages
"""

import time
from fastapi import APIRouter, HTTPException

from app.models.model_manager import model_manager
from app.schemas.moderation import (
    ModerationRequest,
    ModerationResponse,
    BatchModerationRequest,
    BatchModerationResponse,
)
from app.services.moderation_service import moderation_service

router = APIRouter(prefix="/api/v1", tags=["Moderation"])


@router.post("/moderate", response_model=ModerationResponse)
async def moderate_message(request: ModerationRequest):
    """
    Classify a message for toxicity and suggest a polite alternative.

    Returns toxicity scores across 6 categories, overall severity,
    detected language, and a suggested rephrasing if the message is toxic.
    """
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")

    try:
        result = await moderation_service.moderate(
            text=request.text,
            context=request.context,
            channel_id=request.channel_id,
            user_id=request.user_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")


@router.post("/moderate/batch", response_model=BatchModerationResponse)
async def moderate_batch(request: BatchModerationRequest):
    """
    Moderate multiple messages in a single request.
    Max 50 messages per batch.
    """
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")

    start_time = time.perf_counter()

    try:
        results = await moderation_service.moderate_batch(
            texts=request.texts,
            channel_id=request.channel_id,
            user_id=request.user_id,
        )
        total_time = int((time.perf_counter() - start_time) * 1000)

        return BatchModerationResponse(
            results=results,
            total_inference_time_ms=total_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch moderation failed: {str(e)}")
