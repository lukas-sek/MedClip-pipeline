"""
POST /users — create a new anonymous user row and return their UUID.

Called once on first app launch. The returned user_id is persisted
client-side and passed in all subsequent /swipe and /feed requests.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession

from MedClip_pipeline.api.schemas import UserResponse
from MedClip_pipeline.db.engine import get_session
from MedClip_pipeline.db.models import User

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    db: AsyncSession = Depends(get_session),
) -> UserResponse:
    """Create a new user with zeroed preference vectors and return their UUID."""
    user = User()
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserResponse(user_id=user.id)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> UserResponse:
    """Check whether a user exists. Returns 404 if not found."""
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(user_id=user.id)
