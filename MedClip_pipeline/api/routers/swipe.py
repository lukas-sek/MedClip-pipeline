"""
POST /swipe — record a swipe and update per-category preference vectors.

Like:    EMA update (α=0.8) on the category vector, L2-normalised before save.
Dislike: append item_id to negative_history only; no vector change.
Both:    append item_id to interacted_item_ids.
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient
from sqlmodel.ext.asyncio.session import AsyncSession

from MedClip_pipeline.api.schemas import SwipeRequest, SwipeResponse
from MedClip_pipeline.db.engine import get_session
from MedClip_pipeline.db.models import User
from MedClip_pipeline.embeddings.vector_store import get_client

router = APIRouter(prefix="/swipe", tags=["swipe"])

_ALPHA: float = 0.8


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm < 1e-10 else v / norm


def apply_ema(current: list[float] | None, incoming: list[float]) -> list[float]:
    """Return L2-normalised EMA of current and incoming vectors."""
    iv = np.array(incoming, dtype="float32")
    if current is None:
        return l2_normalize(iv).tolist()
    cv = np.array(current, dtype="float32")
    return l2_normalize(_ALPHA * cv + (1 - _ALPHA) * iv).tolist()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_qdrant() -> QdrantClient:
    return get_client()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("", response_model=SwipeResponse)
async def swipe(
    body: SwipeRequest,
    db: AsyncSession = Depends(get_session),
) -> SwipeResponse:
    user = await db.get(User, body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.item_id not in user.interacted_item_ids:
        user.interacted_item_ids = user.interacted_item_ids + [body.item_id]

    vector_updated = False

    if body.action == "like":
        current = user.category_vectors.get(body.category)
        updated = apply_ema(current, body.swiped_item_vector)
        user.category_vectors = {**user.category_vectors, body.category: updated}
        vector_updated = True
    else:
        if body.item_id not in user.negative_history:
            user.negative_history = user.negative_history + [body.item_id]

    db.add(user)
    await db.commit()

    return SwipeResponse(
        user_id=body.user_id,
        item_id=body.item_id,
        category=body.category,
        action=body.action,
        vector_updated=vector_updated,
    )
