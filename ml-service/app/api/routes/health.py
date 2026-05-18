"""
SafeChat — Health Check API Route

GET /api/v1/health — Service health with model status and GPU info
GET /api/v1/ready — Readiness probe for load balancers and container orchestrators
"""

import time
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.model_manager import model_manager

router = APIRouter(prefix="/api/v1", tags=["Health"])

_start_time = time.time()


@router.get("/health")
async def health_check():
    """
    Comprehensive health check including model status and GPU metrics.

    Used by:
      - Spring Boot backend to verify ML service availability
      - Docker health checks
      - Monitoring dashboards
    """
    import torch

    health = model_manager.get_health()

    # Add GPU memory info if available
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
        }

    return {
        **health,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": int(time.time() - _start_time),
        "gpu": gpu_info,
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — returns 200 only when models are loaded and ready.

    Returns HTTP 503 when not ready so that load balancers, Docker, and
    Kubernetes correctly stop routing traffic to this instance.
    """
    if model_manager.is_ready:
        return {"ready": True}

    raise HTTPException(
        status_code=503,
        detail="Models still loading. Service is not ready to accept requests.",
    )

