from __future__ import annotations

import os
import uuid
from collections.abc import Sequence

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "biomedical_embeddings")
VECTOR_SIZE = 512
_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


def get_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        check_compatibility=False,
    )


async def ensure_collection(client: AsyncQdrantClient) -> None:
    collections = await client.get_collections()
    if COLLECTION_NAME not in {collection.name for collection in collections.collections}:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _point_id(filename: str, index: int) -> str:
    # Qdrant accepts unsigned ints or UUIDs only (not raw hex digests).
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{filename}:{index}"))


async def upsert_embeddings(
    client: AsyncQdrantClient,
    embeddings: np.ndarray,
    metadata: Sequence[dict[str, str | int | float | None]],
) -> None:
    if len(embeddings) != len(metadata):
        raise ValueError("Each embedding must have one metadata record")
    points = [
        PointStruct(
            id=_point_id(str(item.get("filename", "unknown")), index),
            vector=vector.tolist(),
            payload=dict(item),
        )
        for index, (vector, item) in enumerate(zip(embeddings, metadata))
    ]
    if points:
        await client.upsert(collection_name=COLLECTION_NAME, points=points)


async def search_embeddings(
    client: AsyncQdrantClient, query_vector: np.ndarray, limit: int
) -> list[dict]:
    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )
    return [
        {"image_path": point.payload.get("image_path"), "score": point.score}
        for point in response.points
    ]