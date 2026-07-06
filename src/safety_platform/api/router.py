"""
SafeChat API Gateway — Main Router

Aggregates all API sub-routers and real-time WebSocket endpoints.
"""

from fastapi import APIRouter
from safety_platform.api.routes import auth, channels, chat, moderation, health, websocket

api_router = APIRouter()

# REST API v1 Routes
api_router.include_router(auth.router)
api_router.include_router(channels.router)
api_router.include_router(chat.router)
api_router.include_router(moderation.router)
api_router.include_router(health.router)
