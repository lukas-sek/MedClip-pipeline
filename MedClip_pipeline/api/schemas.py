import uuid
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel


class SwipeRequest(BaseModel):
    user_id: uuid.UUID
    item_id: int
    category: str
    swiped_item_vector: list[float]
    action: Literal["like", "dislike"]


class SwipeResponse(BaseModel):
    user_id: uuid.UUID
    item_id: int
    category: str
    action: Literal["like", "dislike"]
    vector_updated: bool


class FeedItem(BaseModel):
    id: int
    title: str
    price: Decimal
    currency: str
    brand: Optional[str]
    size: Optional[str]
    image_url: str
    category: str
    vector: list[float]
    source: Literal["exploit", "explore"]


class FeedResponse(BaseModel):
    user_id: uuid.UUID
    items: list[FeedItem]


class UserResponse(BaseModel):
    user_id: uuid.UUID
