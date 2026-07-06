"""
SafeChat API Gateway — ML Service Client

Asynchronous HTTP client for communicating with the Python ML inference service
(fine-tuned MuRIL classifier and Gemini LLM detoxifier).
"""

import httpx
from typing import Dict, Any, Optional, List
from loguru import logger
from safety_platform.core.config import settings


class MLClientService:
    """Async client communicating with SafeChat ML Service (Port 8001 / 8000)."""

    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL.rstrip("/")
        self.timeout = settings.ML_SERVICE_TIMEOUT_SECONDS

    async def moderate_text(self, text: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send text to ML service for toxicity classification with context snapshot.
        Returns classification dictionary or safe fallback if ML service is unreachable.
        """
        url = f"{self.base_url}/api/v1/moderate"
        payload = {"text": text, "context": context or []}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json()
                logger.warning(f"ML Service returned HTTP {res.status_code}: {res.text}")
        except httpx.RequestError as e:
            logger.error(f"ML Service connection failed at {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling ML Service: {e}")

        # Fallback response so chat remains functional
        return {
            "is_toxic": False,
            "overall_score": 0.0,
            "severity": "SAFE",
            "categories": {},
            "suggestion": None,
            "detected_language": "en",
            "inference_time_ms": 0,
            "model_version": "fallback-offline",
        }

    async def detoxify_text(self, text: str, target_language: Optional[str] = "en", context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Request intent-preserving LLM style transfer from ML Service."""
        url = f"{self.base_url}/api/v1/detoxify"
        payload = {"text": text, "target_language": target_language, "context": context or []}

        try:
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.error(f"Detoxification call failed: {e}")

        return {
            "original_text": text,
            "detoxified_text": "Please use respectful and constructive language.",
            "language": target_language or "en",
            "method": "template-fallback",
        }

    async def check_ml_health(self) -> Dict[str, Any]:
        """Check health of remote ML service."""
        url = f"{self.base_url}/api/v1/health"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return {"status": "ok", **res.json()}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
        return {"status": "unreachable"}


ml_client = MLClientService()
