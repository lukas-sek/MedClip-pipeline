"""
MedClip-pipeline FastAPI application.

Run locally:
    uv run uvicorn MedClip_pipeline.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from MedClip_pipeline.api.routers import feed, swipe, users
from MedClip_pipeline.db.engine import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Create database tables on startup."""
    await init_db()
    yield


app = FastAPI(
    title="MedClip-pipeline",
    description="AI-powered vintage fashion recommendation engine.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(users.router)
app.include_router(swipe.router)
app.include_router(feed.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
