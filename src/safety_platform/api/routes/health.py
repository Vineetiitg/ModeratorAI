"""
SafeChat API Gateway — Comprehensive Health & Readiness Probes

GET /api/v1/health — Verifies status of PostgreSQL, Redis, MongoDB, and ML Service
GET /api/v1/ready — Kubernetes / Docker readiness probe
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from safety_platform.core.db import get_db, get_redis, get_mongo_db
from safety_platform.services.ml_client import ml_client
from safety_platform.core.config import settings

router = APIRouter(tags=["Health & Readiness Probes"])
_start_time = time.time()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Comprehensive polyglot health check:
    1. PostgreSQL (Relational Core)
    2. Redis (Rate Limiter & PubSub Cache)
    3. MongoDB (Unstructured Chat Store & Feedback)
    4. ML Service (Fine-tuned MuRIL & Gemini AI)
    """
    status_report: Dict[str, Any] = {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": int(time.time() - _start_time),
        "engines": {},
    }

    # 1. Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        status_report["engines"]["postgresql"] = "ok"
    except Exception as e:
        status_report["engines"]["postgresql"] = f"error: {str(e)[:50]}"

    # 2. Check Redis
    redis_client = get_redis()
    if redis_client:
        try:
            await redis_client.ping()
            status_report["engines"]["redis"] = "ok"
        except Exception as e:
            status_report["engines"]["redis"] = f"error: {str(e)[:50]}"
    else:
        status_report["engines"]["redis"] = "offline (in-memory fallback)"

    # 3. Check MongoDB
    mongo_db = get_mongo_db()
    if mongo_db is not None:
        try:
            await mongo_db.client.admin.command("ping")
            status_report["engines"]["mongodb"] = "ok"
        except Exception as e:
            status_report["engines"]["mongodb"] = f"error: {str(e)[:50]}"
    else:
        status_report["engines"]["mongodb"] = "offline (in-memory fallback)"

    # 4. Check ML Service
    ml_health = await ml_client.check_ml_health()
    status_report["engines"]["ml_service"] = ml_health.get("status", "unknown")

    # Determine overall status
    engines_status = list(status_report["engines"].values())
    if any(s.startswith("error") for s in engines_status):
        status_report["overall_status"] = "degraded"
    else:
        status_report["overall_status"] = "healthy"

    return status_report


@router.get("/ready")
async def readiness_probe(db: AsyncSession = Depends(get_db)) -> Dict[str, str]:
    """Readiness probe checking PostgreSQL connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": "true"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not reachable. Gateway is not ready."
        )
