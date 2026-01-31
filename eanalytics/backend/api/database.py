from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.sql import func
from typing import Optional
import redis.asyncio as redis
from .config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Redis connection
redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    redis_client = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await redis_client.ping()
    return redis_client

async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if redis_client is None:
        await init_redis()
    return redis_client

# Base model
class Base(DeclarativeBase):
    pass

# Dependency
async def get_db():
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # postgresql, mysql, mongodb, salesforce, etc.
    connection_config = Column(JSON, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    last_connected = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
class QueryExecution(Base):
    __tablename__ = "query_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Integer, nullable=True)  # in milliseconds
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Dashboard(Base):
    __tablename__ = "dashboards"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    layout = Column(JSON, nullable=False, default=dict)
    is_public = Column(Boolean, default=False)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Visualization(Base):
    __tablename__ = "visualizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # line, bar, pie, scatter, etc.
    dashboard_id = Column(Integer, ForeignKey("dashboards.id"), nullable=False)
    query_id = Column(Integer, ForeignKey("query_executions.id"), nullable=True)
    config = Column(JSON, nullable=False, default=dict)
    data = Column(JSON, nullable=True)
    position = Column(JSON, nullable=False, default=dict)  # {x, y, w, h}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())