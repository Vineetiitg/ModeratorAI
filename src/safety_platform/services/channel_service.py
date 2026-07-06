"""
SafeChat API Gateway — Channel Service (PostgreSQL)

Handles chat room creation, listing, and automatic seeding of the default
'# general' channel in PostgreSQL on startup.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from safety_platform.models.postgres import Channel, User
from safety_platform.schemas.channel import ChannelCreateRequest


class ChannelService:
    """Service handling discussion channel operations in PostgreSQL."""

    @staticmethod
    async def get_all_channels(db: AsyncSession) -> List[Channel]:
        """Retrieve all active chat channels."""
        stmt = select(Channel).order_by(Channel.name.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def create_channel(db: AsyncSession, payload: ChannelCreateRequest, creator: Optional[User] = None) -> Channel:
        """Create a new chat channel in PostgreSQL."""
        name_clean = payload.name.strip().lower()
        if name_clean.startswith("#"):
            name_clean = name_clean[1:].strip()

        stmt = select(Channel).where(Channel.name == name_clean)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            return existing

        channel = Channel(
            name=name_clean,
            description=payload.description or f"Discussion channel for #{name_clean}",
            created_by_id=creator.id if creator else None,
        )
        db.add(channel)
        await db.commit()
        await db.refresh(channel)
        logger.info(f"Created channel #{channel.name} in PostgreSQL (ID: {channel.id})")
        return channel

    @staticmethod
    async def seed_default_channels(db: AsyncSession) -> None:
        """Seed default channels (# general, # hindi-discussion) on startup."""
        default_rooms = [
            {"name": "general", "description": "Welcome to general multilingual discussion."},
            {"name": "hindi-en", "description": "Hinglish & Devanagari code-mixed chat space."},
        ]
        for room in default_rooms:
            stmt = select(Channel).where(Channel.name == room["name"])
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()
            if not existing:
                ch = Channel(name=room["name"], description=room["description"])
                db.add(ch)
                logger.info(f"Seeding default room: #{room['name']}")
        await db.commit()
