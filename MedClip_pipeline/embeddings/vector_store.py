"""
Qdrant collection management and vector upsert logic.
Uses HTTP REST transport for compatibility across client/server versions.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "vintage_garments"
VECTOR_SIZE = 768
_QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_HTTP_PORT: int = int(os.getenv("QDRANT_HTTP_PORT", "6333"))


@dataclass
class EmbedRecord:
    """Holds a product ID, its embedding vector, and metadata payload."""

    product_id: int
    vector: np.ndarray
    brand: str | None
    price: str
    size: str | None
    image_url: str
    category: str = "unknown"


def get_client() -> QdrantClient:
    """Return a Qdrant client connected over HTTP REST."""
    return QdrantClient(
        host=_QDRANT_HOST,
        port=_QDRANT_HTTP_PORT,
        prefer_grpc=False,
        check_compatibility=False,
    )


def init_collection(client: QdrantClient) -> None:
    """
    Ensure the `vintage_garments` collection exists with the correct schema.
    No-op if the collection already exists.
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        logger.info(
            "Collection '%s' already exists — skipping creation.", COLLECTION_NAME
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info(
        "Created Qdrant collection '%s' (size=%s, distance=COSINE).",
        COLLECTION_NAME,
        VECTOR_SIZE,
    )


def upsert_batch(client: QdrantClient, records: list[EmbedRecord]) -> None:
    """
    Upsert a batch of embedding records into Qdrant over HTTP REST.
    Each point carries the vector plus product metadata as payload.
    """
    if not records:
        return

    points = [
        PointStruct(
            id=record.product_id,
            vector=record.vector.tolist(),
            payload={
                "brand": record.brand,
                "price": record.price,
                "size": record.size,
                "image_url": record.image_url,
                "category": record.category,
            },
        )
        for record in records
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Upserted %s vectors to '%s'.", len(points), COLLECTION_NAME)
