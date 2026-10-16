import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

load_dotenv()

_engine: AsyncEngine | None = None


def _get_engine() -> AsyncEngine:
    """Return the shared async engine, creating it on first call."""
    global _engine
    if _engine is None:
        url = os.environ["DATABASE_URL"]
        _engine = create_async_engine(url, echo=False, future=True)
    return _engine


async def init_db() -> None:
    """Create all tables defined in SQLModel metadata.

    Set RESET_DB=true to drop all tables first (schema migration for dev).
    """
    async with _get_engine().begin() as conn:
        if os.getenv("RESET_DB", "").lower() == "true":
            await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with AsyncSession(_get_engine(), expire_on_commit=False) as session:
        yield session
