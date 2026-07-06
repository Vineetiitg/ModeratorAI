"""
SafeChat API Gateway — Chat History Routes (MongoDB)

GET /api/v1/chat/messages/{channel_id} — Retrieve recent chat messages from MongoDB
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Path, Query

from safety_platform.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat History (MongoDB)"])


@router.get("/messages/{channel_id}", response_model=List[Dict[str, Any]])
async def get_channel_history(
    channel_id: str = Path(..., description="Target channel name or ID"),
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
):
    """Fetch recent chat messages for a channel from MongoDB."""
    return await ChatService.get_recent_messages(channel_id=channel_id, limit=limit)
