"""
SafeChat API Gateway — Channel Routes (PostgreSQL)

GET /api/v1/channels — List active chat rooms from PostgreSQL
POST /api/v1/channels — Create a new chat room
"""

from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from safety_platform.core.db import get_db
from safety_platform.core.security import get_current_user_optional
from safety_platform.models.postgres import User
from safety_platform.schemas.channel import ChannelCreateRequest, ChannelResponse
from safety_platform.services.channel_service import ChannelService

router = APIRouter(prefix="/channels", tags=["Channels (PostgreSQL)"])


@router.get("", response_model=List[ChannelResponse])
async def list_channels(db: AsyncSession = Depends(get_db)):
    """List all available chat rooms from PostgreSQL."""
    return await ChannelService.get_all_channels(db)


@router.post("", response_model=ChannelResponse, status_code=201)
async def create_channel(
    payload: ChannelCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new chat room in PostgreSQL."""
    return await ChannelService.create_channel(db, payload, creator=current_user)
