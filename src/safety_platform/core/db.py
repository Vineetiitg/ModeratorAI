"""
SafeChat API Gateway — Polyglot Database Connections

Manages connection lifecycle and dependency injection for:
1. PostgreSQL (via Async SQLAlchemy & Asyncpg) — Relational Core
2. Redis (via redis.asyncio) — High-Speed Caching & Rate Limiting
3. MongoDB (via Motor Async Driver) — Chat History & ML Traces
"""

from typing import AsyncGenerator, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from safety_platform.core.config import settings

# ── 1. PostgreSQL Setup (SQLAlchemy Async Engine) ──────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an async PostgreSQL database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_postgres_tables() -> None:
    """Create all relational tables in PostgreSQL on startup if they don't exist."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.success("PostgreSQL tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL tables: {e}")


# ── 2. Redis Setup (Async Pool) ────────────────────────────────────────
_redis_pool: Optional[redis.Redis] = None


async def connect_redis() -> None:
    """Initialize Redis connection pool during startup."""
    global _redis_pool
    try:
        _redis_pool = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
        )
        await _redis_pool.ping()
        logger.success(f"Connected to Redis at {settings.REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Rate limiting will fall back to memory.")
        _redis_pool = None


async def disconnect_redis() -> None:
    """Close Redis connection pool during shutdown."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.close()
        logger.info("Redis connection closed.")


def get_redis() -> Optional[redis.Redis]:
    """Get active Redis client instance."""
    return _redis_pool


# ── 3. MongoDB Setup (Motor Async Driver) ──────────────────────────────
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None


async def connect_mongo() -> None:
    """Initialize MongoDB connection during startup."""
    global _mongo_client, _mongo_db
    try:
        _mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
        _mongo_db = _mongo_client[settings.MONGODB_DB]
        await _mongo_client.admin.command("ping")
        logger.success(f"Connected to MongoDB at {settings.MONGODB_URL}/{settings.MONGODB_DB}")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Chat storage may fall back to in-memory.")
        _mongo_client = None
        _mongo_db = None


async def disconnect_mongo() -> None:
    """Close MongoDB connection during shutdown."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        logger.info("MongoDB connection closed.")


def get_mongo_db() -> Optional[AsyncIOMotorDatabase]:
    """Get active MongoDB database instance."""
    return _mongo_db
