from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import create_engine
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/rockwool_tracker"
)
SYNC_DATABASE_URL = os.getenv(
    "SYNC_DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rockwool_tracker"
)

# Async engine for FastAPI
async_engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# Sync engine for Alembic migrations
sync_engine = create_engine(SYNC_DATABASE_URL)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    from .models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
