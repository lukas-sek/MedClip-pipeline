"""
GET /feed — return 20 personalised items.

All feed (category=None or "all"):
    Per-category Qdrant recommend queries batched into a single round-trip.
    Each active category in the user's preference dictionary contributes an
    equal share of the exploit budget (16 items), padded with PostgreSQL
    exploration.

Single-category feed (category=<name>):
    Warm (user has a vector for this category): single recommend query,
    limit=20, filtered to that category.
    Cold (no vector yet): Qdrant scroll filtered to that category, limit=20.
    PostgreSQL exploration is also filtered to the same category.
"""

import random
import uuid
from math import ceil
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    QueryRequest,
    RecommendInput,
    RecommendQuery,
)
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from MedClip_pipeline.api.routers.swipe import get_qdrant
from MedClip_pipeline.api.schemas import FeedItem, FeedResponse
from MedClip_pipeline.db.engine import get_session
from MedClip_pipeline.db.models import Product, User
from MedClip_pipeline.embeddings.vector_store import COLLECTION_NAME

router = APIRouter(prefix="/feed", tags=["feed"])

_TOTAL_ITEMS = 20
_EXPLOIT_BUDGET = 16


# ---------------------------------------------------------------------------
# Qdrant request builders
# ---------------------------------------------------------------------------


def _build_batch_requests(
    category_vectors: dict[str, list[float]],
    exclude_ids: list[int],
) -> list[QueryRequest]:
    """One QueryRequest per active category, budget split evenly."""
    if not category_vectors:
        return []

    per_cat = ceil(_EXPLOIT_BUDGET / len(category_vectors))
    exclude_filter = HasIdCondition(has_id=exclude_ids) if exclude_ids else None

    requests = []
    for category, vector in category_vectors.items():
        must_not = [exclude_filter] if exclude_filter else []
        requests.append(
            QueryRequest(
                query=RecommendQuery(recommend=RecommendInput(positive=[vector])),
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category),
                        )
                    ],
                    must_not=must_not,
                ),
                limit=per_cat,
                with_payload=True,
                with_vector=True,
            )
        )
    return requests


def _build_single_warm_request(
    category: str,
    vector: list[float],
    exclude_ids: list[int],
) -> QueryRequest:
    """Single recommend query for a category where the user has a preference vector."""
    exclude_filter = HasIdCondition(has_id=exclude_ids) if exclude_ids else None
    must_not = [exclude_filter] if exclude_filter else []
    return QueryRequest(
        query=RecommendQuery(recommend=RecommendInput(positive=[vector])),
        filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))],
            must_not=must_not,
        ),
        limit=_TOTAL_ITEMS,
        with_payload=True,
        with_vector=True,
    )


def _build_cold_scroll_filter(category: str, exclude_ids: list[int]) -> Filter:
    """Scroll filter for cold-start single-category feed."""
    exclude_filter = HasIdCondition(has_id=exclude_ids) if exclude_ids else None
    must_not = [exclude_filter] if exclude_filter else []
    return Filter(
        must=[FieldCondition(key="category", match=MatchValue(value=category))],
        must_not=must_not,
    )


# ---------------------------------------------------------------------------
# Shared result mapper
# ---------------------------------------------------------------------------


def _point_to_feed_item(point, seen_ids: set[int]) -> FeedItem | None:
    if point.id in seen_ids:
        return None
    payload = point.payload or {}
    vector = point.vector if isinstance(point.vector, list) else []
    return FeedItem(
        id=int(point.id),
        title=payload.get("title", ""),
        price=payload.get("price", "0"),
        currency=payload.get("currency", ""),
        brand=payload.get("brand"),
        size=payload.get("size"),
        image_url=payload.get("image_url", ""),
        category=payload.get("category", "unknown"),
        vector=vector,
        source="exploit",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("", response_model=FeedResponse)
async def feed(
    user_id: uuid.UUID,
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_session),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> FeedResponse:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    seen_ids: set[int] = set(user.interacted_item_ids)
    exploit_items: list[FeedItem] = []

    # Normalise the category filter.
    cat_filter: str | None = None
    if category and category.strip().lower() not in ("", "all"):
        cat_filter = category.strip().lower()

    if cat_filter:
        # ------------------------------------------------------------------
        # Single-category feed
        # ------------------------------------------------------------------
        cat_vector = user.category_vectors.get(cat_filter)

        try:
            if cat_vector:
                # Warm path: personalised recommendation within this category.
                request = _build_single_warm_request(
                    cat_filter, cat_vector, user.interacted_item_ids
                )
                results = qdrant.query_batch_points(
                    collection_name=COLLECTION_NAME,
                    requests=[request],
                )
                for point in results[0].points:
                    item = _point_to_feed_item(point, seen_ids)
                    if item:
                        exploit_items.append(item)
                        seen_ids.add(item.id)
            else:
                # Cold path: scroll the category to seed the user's taste.
                scroll_filter = _build_cold_scroll_filter(
                    cat_filter, user.interacted_item_ids
                )
                scroll_result = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=_TOTAL_ITEMS,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in scroll_result[0]:
                    item = _point_to_feed_item(point, seen_ids)
                    if item:
                        exploit_items.append(item)
                        seen_ids.add(item.id)
        except Exception:
            pass

        # PostgreSQL exploration fallback, filtered to the same category.
        explore_needed = _TOTAL_ITEMS - len(exploit_items)
        if explore_needed > 0:
            result = await db.exec(
                select(Product).where(
                    Product.embedded_at.is_not(None),
                    Product.category == cat_filter,
                )
            )
            candidates = [p for p in result.scalars().all() if p.id not in seen_ids]
            random.shuffle(candidates)
            exploit_items += [
                FeedItem(
                    id=p.id,
                    title=p.title,
                    price=p.price,
                    currency=p.currency,
                    brand=p.brand,
                    size=p.size,
                    image_url=p.image_url,
                    category=p.category,
                    vector=[],
                    source="explore",
                )
                for p in candidates[:explore_needed]
            ]

    else:
        # ------------------------------------------------------------------
        # All-categories feed (existing logic)
        # ------------------------------------------------------------------
        if user.category_vectors:
            try:
                batch_requests = _build_batch_requests(
                    user.category_vectors,
                    exclude_ids=user.interacted_item_ids,
                )
                batch_results = qdrant.query_batch_points(
                    collection_name=COLLECTION_NAME,
                    requests=batch_requests,
                )
                for group in batch_results:
                    for point in group.points:
                        item = _point_to_feed_item(point, seen_ids)
                        if item:
                            exploit_items.append(item)
                            seen_ids.add(item.id)
            except Exception:
                pass

        # PostgreSQL exploration — no category filter.
        explore_needed = _TOTAL_ITEMS - len(exploit_items)
        result = await db.exec(select(Product).where(Product.embedded_at.is_not(None)))
        all_products = [p for p in result.scalars().all() if p.id not in seen_ids]
        random.shuffle(all_products)
        exploit_items += [
            FeedItem(
                id=p.id,
                title=p.title,
                price=p.price,
                currency=p.currency,
                brand=p.brand,
                size=p.size,
                image_url=p.image_url,
                category=p.category,
                vector=[],
                source="explore",
            )
            for p in all_products[:explore_needed]
        ]

    random.shuffle(exploit_items)
    return FeedResponse(user_id=user_id, items=exploit_items[:_TOTAL_ITEMS])
