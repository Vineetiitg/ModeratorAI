"""
SafeChat ML Service — FastAPI Application

Entry point for the ML inference service.
Handles toxicity classification, LLM-powered detoxification,
real-time WebSocket moderation, and feedback collection.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Or for production:
    gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    (Use 1 worker since models are loaded in-memory per worker)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.models.model_manager import model_manager

# Import route modules
from app.api.routes import moderation, detoxify, health, feedback, websocket


# ── Application Lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load ML models, initialize LLM client, connect to MongoDB.
    Shutdown: Clean up GPU resources, close DB connections.
    """
    from app.services.feedback_service import feedback_service

    # ── STARTUP ────────────────────────────────────────
    logger.info("Starting SafeChat ML Service...")
    await feedback_service.connect()
    await model_manager.initialize()
    logger.success(f"SafeChat ML Service ready on {settings.HOST}:{settings.PORT}")

    yield  # Application runs here

    # ── SHUTDOWN ───────────────────────────────────────
    logger.info("Shutting down SafeChat ML Service...")
    await feedback_service.disconnect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete. Goodbye!")


# ── Create FastAPI Application ─────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time multilingual toxicity classification and LLM-powered detoxification service. "
        "Supports English, Hindi, Hinglish (code-mixed), and Indian languages. "
        "Features a fine-tuned HingBERT classifier with context-aware prediction "
        "and Gemini-based intent-preserving style transfer."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS Middleware ────────────────────────────────────────────────────
# Origins read from settings (configurable via env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register Routes ───────────────────────────────────────────────────

app.include_router(moderation.router)
app.include_router(detoxify.router)
app.include_router(health.router)
app.include_router(feedback.router)
app.include_router(websocket.router)  # WebSocket for real-time chat


# ── Root Endpoint ─────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Service info and links to documentation."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "moderate": "POST /api/v1/moderate",
            "moderate_batch": "POST /api/v1/moderate/batch",
            "detoxify": "POST /api/v1/detoxify",
            "feedback": "POST /api/v1/feedback",
            "feedback_stats": "GET /api/v1/feedback/stats",
            "health": "GET /api/v1/health",
            "ready": "GET /api/v1/ready",
            "websocket": "WS /ws/chat",
        },
    }

