"""
SafeChat ML Service — Configuration

All settings can be overridden via environment variables prefixed with SAFECHAT_.
Example: SAFECHAT_GEMINI_API_KEY=your_key
"""

from pydantic_settings import BaseSettings
from typing import List
import torch


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "SafeChat ML Service"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # ── Device ───────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Toxicity Classifier (fine-tuned Hing-RoBERTa) ───────────
    # Points to local fine-tuned checkpoint by default.
    # Override with SAFECHAT_CLASSIFIER_MODEL for hub models.
    CLASSIFIER_MODEL: str = "./checkpoints/hingbert-toxicity-finetuned"

    # ── LLM Detoxification (Gemini API) ──────────────────
    # Replaces the old IndicBART approach with LLM-powered
    # intent-preserving style transfer via Gemini.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    DETOX_MAX_TOKENS: int = 256

    # ── Inference Settings ───────────────────────────────
    MAX_SEQ_LENGTH: int = 256
    BATCH_SIZE: int = 8

    # ── Severity Thresholds ──────────────────────────────
    THRESHOLD_SAFE: float = 0.30
    THRESHOLD_LOW: float = 0.55
    THRESHOLD_MEDIUM: float = 0.75

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]

    # ── MongoDB & Continuous Learning ──────────────────────
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "safechat"
    FEEDBACK_COLLECTION: str = "feedback"
    FEEDBACK_THRESHOLD_FOR_RETRAIN: int = 500
    MODEL_CHECKPOINT_DIR: str = "./checkpoints"

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# Singleton settings instance
settings = Settings()
