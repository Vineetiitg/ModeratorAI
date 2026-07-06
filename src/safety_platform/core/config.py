"""
SafeChat API Gateway — Core Configuration

Loads environment variables using pydantic-settings for polyglot persistence
(PostgreSQL, Redis, MongoDB), JWT authentication, and ML service communication.
"""

from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallback."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App Metadata
    APP_NAME: str = "SafeChat API Gateway"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api/v1"

    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # PostgreSQL (Relational Core)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://safety_user:safety_password@localhost:5432/safety_platform",
        description="Async SQLAlchemy URL for PostgreSQL"
    )

    # Redis (Rate Limiter & Cache)
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    RATE_LIMIT_MESSAGES_PER_MINUTE: int = 20

    # MongoDB (Chat History & Continuous Learning Store)
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URL"
    )
    MONGODB_DB: str = "safechat"
    CHAT_COLLECTION: str = "chat_messages"

    # ML Service Communication
    ML_SERVICE_URL: str = Field(
        default="http://localhost:8001",
        description="URL of the fine-tuned MuRIL & Gemini ML service"
    )
    ML_SERVICE_TIMEOUT_SECONDS: float = 10.0

    # JWT Authentication
    JWT_SECRET_KEY: str = Field(
        default="supersecretkey_change_in_production_998877665544332211",
        description="Secret key for signing JWT tokens"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # CORS Policy
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
