"""
SafeChat API Gateway — Chat & WebSocket Orchestration Service

Handles:
1. Redis sliding-window rate limiting to prevent spam and DDoS
2. Persistent unstructured message and AI trace logging in MongoDB
3. WebSocket connection management and real-time broadcasting
"""

import time
import json
import uuid
from typing import Dict, List, Optional, Set
from fastapi import WebSocket
from loguru import logger

from safety_platform.core.config import settings
from safety_platform.core.db import get_redis, get_mongo_db
from safety_platform.models.mongo import ChatMessageDocument, ModerationMetadata


class ChatService:
    """Orchestrates chat persistence in MongoDB and rate limiting in Redis."""

    @staticmethod
    async def check_rate_limit(user_id: str) -> bool:
        """
        Check if user exceeded RATE_LIMIT_MESSAGES_PER_MINUTE using Redis.
        Returns True if allowed, False if rate limited.
        """
        redis_client = get_redis()
        if not redis_client:
            return True  # Fall back to allow if Redis is offline

        key = f"rate_limit:chat:{user_id}"
        now = int(time.time())
        window_start = now - 60

        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                # Remove timestamps older than 60 seconds
                pipe.zremrangebyscore(key, 0, window_start)
                # Count current messages in window
                pipe.zcard(key)
                # Add current message timestamp
                pipe.zadd(key, {str(now) + "-" + str(uuid.uuid4())[:8]: now})
                # Set key expiry to clean up memory
                pipe.expire(key, 120)
                results = await pipe.execute()

            current_count = results[1]
            if current_count >= settings.RATE_LIMIT_MESSAGES_PER_MINUTE:
                logger.warning(f"User {user_id} rate limited ({current_count} msgs/min)")
                return False
            return True
        except Exception as e:
            logger.error(f"Redis rate limit check error: {e}")
            return True

    @staticmethod
    async def save_message_to_mongo(
        channel_id: str,
        sender_id: str,
        sender_name: str,
        content: str,
        status: str,
        moderation_data: Optional[Dict] = None,
        context_snapshot: Optional[List[str]] = None,
    ) -> ChatMessageDocument:
        """Save moderated chat message document into MongoDB."""
        msg_id = str(uuid.uuid4())
        mod_meta = ModerationMetadata(**moderation_data) if moderation_data else None

        doc = ChatMessageDocument(
            message_id=msg_id,
            channel_id=channel_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            status=status,
            moderation=mod_meta,
            context_snapshot=context_snapshot or [],
        )

        mongo_db = get_mongo_db()
        if mongo_db is not None:
            try:
                collection = mongo_db[settings.CHAT_COLLECTION]
                await collection.insert_one(doc.to_mongo_dict())
                logger.info(f"Saved message {msg_id} to MongoDB ({channel_id})")
            except Exception as e:
                logger.error(f"MongoDB save failed for message {msg_id}: {e}")
        else:
            logger.warning(f"MongoDB offline: message {msg_id} kept in memory only.")

        return doc

    @staticmethod
    async def get_recent_messages(channel_id: str, limit: int = 50) -> List[Dict]:
        """Fetch recent messages for a channel from MongoDB."""
        mongo_db = get_mongo_db()
        if mongo_db is None:
            return []

        try:
            collection = mongo_db[settings.CHAT_COLLECTION]
            cursor = collection.find(
                {"channel_id": channel_id},
                {"_id": 0}
            ).sort("created_at", 1).limit(limit)

            docs = await cursor.to_list(length=limit)
            # Map message_id to id for frontend compatibility
            for d in docs:
                if "message_id" in d and "id" not in d:
                    d["id"] = d["message_id"]
            return docs
        except Exception as e:
            logger.error(f"MongoDB fetch failed for channel {channel_id}: {e}")
            return []


class WebSocketConnectionManager:
    """Manages active WebSocket client connections per channel."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_mapping: Dict[WebSocket, Dict[str, str]] = {}

    async def connect(self, websocket: WebSocket, channel_id: str, user_id: str, user_name: str) -> None:
        """Accept connection and add to channel pool."""
        await websocket.accept()
        if channel_id not in self.active_connections:
            self.active_connections[channel_id] = set()
        self.active_connections[channel_id].add(websocket)
        self.user_mapping[websocket] = {"user_id": user_id, "user_name": user_name, "channel_id": channel_id}
        logger.info(f"WebSocket connected: {user_name} ({user_id}) -> #{channel_id}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove connection from pool."""
        meta = self.user_mapping.pop(websocket, None)
        if meta:
            channel_id = meta["channel_id"]
            if channel_id in self.active_connections:
                self.active_connections[channel_id].discard(websocket)
                if not self.active_connections[channel_id]:
                    del self.active_connections[channel_id]
            logger.info(f"WebSocket disconnected: {meta['user_name']}")

    async def broadcast_to_channel(self, channel_id: str, message_data: Dict) -> None:
        """Broadcast a message payload to all clients connected to channel_id."""
        if channel_id not in self.active_connections:
            return

        payload = json.dumps(message_data, default=str)
        dead_sockets = set()
        for ws in self.active_connections[channel_id]:
            try:
                await ws.send_text(payload)
            except Exception as e:
                logger.error(f"Broadcast error to WS: {e}")
                dead_sockets.add(ws)

        for ws in dead_sockets:
            self.disconnect(ws)


ws_manager = WebSocketConnectionManager()
