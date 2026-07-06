"""
SafeChat — Model Manager
Centralized lifecycle manager for ML models.
"""

from typing import Dict, Optional
import torch
from loguru import logger

from app.config import settings
from app.models.toxicity_classifier import ToxicityClassifier
from app.models.llm_detoxifier import LLMDetoxifier


class ModelManager:
    """Singleton-style manager for toxicity classification and LLM detoxification."""

    def __init__(self):
        self.classifier: Optional[ToxicityClassifier] = None
        self.detoxifier: Optional[LLMDetoxifier] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load all models. Called once during FastAPI startup."""
        logger.info("=" * 60)
        logger.info("  SafeChat Model Manager — Initializing")
        logger.info("=" * 60)
        self._log_hardware_info()

        # Load fine-tuned MuRIL classifier
        try:
            self.classifier = ToxicityClassifier(
                model_name=settings.CLASSIFIER_MODEL,
                device=settings.DEVICE,
            )
            self.classifier.load()
        except Exception as e:
            logger.error(f"Failed to load toxicity classifier: {e}")
            raise RuntimeError(f"Classifier initialization failed: {e}")

        # Initialize LLM Detoxifier (Gemini API — no large model to load)
        try:
            self.detoxifier = LLMDetoxifier()
            if self.detoxifier.is_available:
                logger.success("LLM Detoxifier ready (Gemini API)")
            else:
                logger.warning("LLM Detoxifier running in fallback mode (no API key)")
        except Exception as e:
            logger.warning(f"Detoxifier initialization failed: {e}. Fallback templates will be used.")
            self.detoxifier = LLMDetoxifier()  # Will default to fallback mode

        self._initialized = True
        logger.success("All models initialized successfully!")
        logger.info("=" * 60)

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.classifier is not None and self.classifier.is_loaded

    def get_health(self) -> Dict:
        return {
            "status": "healthy" if self.is_ready else "degraded",
            "models": {
                "toxicity_classifier": self.classifier.get_info() if self.classifier else {"loaded": False},
                "detoxifier": self.detoxifier.get_info() if self.detoxifier else {"mode": "unavailable"},
            },
            "device": settings.DEVICE,
        }

    async def swap_classifier(self, new_model_path: str) -> bool:
        """Hot-swap the toxicity classifier (used after retraining)."""
        logger.info(f"Hot-swapping classifier to: {new_model_path}")
        try:
            new_classifier = ToxicityClassifier(model_name=new_model_path, device=settings.DEVICE)
            new_classifier.load()
            old_classifier = self.classifier
            self.classifier = new_classifier
            del old_classifier
            if settings.DEVICE == "cuda":
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Classifier hot-swap failed: {e}.")
            return False

    @staticmethod
    def _log_hardware_info():
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("No GPU detected. Running on CPU (slower inference).")
        logger.info(f"Selected device: {settings.DEVICE}")


model_manager = ModelManager()

