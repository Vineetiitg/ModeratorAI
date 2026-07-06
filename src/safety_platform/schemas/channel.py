"""
SafeChat API Gateway — Channel Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChannelCreateRequest(BaseModel):
    """Payload for creating a new discussion channel."""
    name: str = Field(..., min_length=2, max_length=50, description="Channel name without prefix")
    description: Optional[str] = Field(None, max_length=255, description="Brief channel description")


class ChannelResponse(BaseModel):
    """Channel information DTO."""
    id: str
    name: str
    description: Optional[str] = None
    created_by_id: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True
