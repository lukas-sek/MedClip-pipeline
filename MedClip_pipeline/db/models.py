import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, DateTime, JSON, Column
from sqlmodel import Field, SQLModel


class Product(SQLModel, table=True):
    """
    Stores validated product metadata from the Vinted scraper.
    The primary key maps 1-to-1 to the Vinted item ID.
    """

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    title: str
    price: Decimal = Field(decimal_places=2, max_digits=10)
    currency: str = Field(max_length=10)
    brand: Optional[str] = Field(default=None)
    size: Optional[str] = Field(default=None)
    image_url: str
    category: str = Field(default="unknown")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True)),
    )
    embedded_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class User(SQLModel, table=True):
    """
    Tracks per-user latent preference state.

    category_vectors: maps garment category → 768-dim L2-normalised preference
                      vector, updated via EMA (α=0.8) on each right swipe.
    interacted_item_ids: all seen items (prevents duplicates in /feed).
    negative_history:    left-swiped item IDs (used as Qdrant exclusion filter).
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )
    category_vectors: dict[str, list[float]] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )
    interacted_item_ids: list[int] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )
    negative_history: list[int] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True)),
    )
