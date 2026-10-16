"""
Phase 2 tests — Qdrant collection management and embedding output.

Run with:
    uv run pytest tests/test_embeddings.py -v
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from MedClip_pipeline.embeddings.vector_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EmbedRecord,
    init_collection,
    upsert_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(product_id: int = 1) -> EmbedRecord:
    return EmbedRecord(
        product_id=product_id,
        vector=np.ones(VECTOR_SIZE, dtype="float32"),
        brand="Levi's",
        price="29.99",
        size="M",
        image_url="https://images.vinted.net/item.jpg",
    )


def _mock_client_with_collections(*names: str) -> MagicMock:
    """Return a mock QdrantClient whose get_collections() contains `names`."""
    mock_collection = MagicMock()
    mock_collection.name = names[0] if names else "__nonexistent__"
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [_named_collection(n) for n in names]
    client = MagicMock()
    client.get_collections.return_value = mock_collections_response
    return client


def _named_collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name
    return c


# ---------------------------------------------------------------------------
# Task 2.3 — Collection initialisation
# ---------------------------------------------------------------------------


class TestInitCollection:
    def test_creates_collection_when_missing(self) -> None:
        client = _mock_client_with_collections()  # no collections exist
        init_collection(client)
        client.create_collection.assert_called_once()
        args, kwargs = client.create_collection.call_args
        assert (
            kwargs.get("collection_name") == COLLECTION_NAME
            or args[0] == COLLECTION_NAME
        )

    def test_skips_creation_when_already_exists(self) -> None:
        client = _mock_client_with_collections(COLLECTION_NAME)
        init_collection(client)
        client.create_collection.assert_not_called()

    def test_creates_with_correct_vector_size(self) -> None:
        from qdrant_client.http.models import VectorParams

        client = _mock_client_with_collections()
        init_collection(client)
        _, kwargs = client.create_collection.call_args
        vector_params: VectorParams = kwargs["vectors_config"]
        assert vector_params.size == VECTOR_SIZE

    def test_creates_with_cosine_distance(self) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        client = _mock_client_with_collections()
        init_collection(client)
        _, kwargs = client.create_collection.call_args
        vector_params: VectorParams = kwargs["vectors_config"]
        assert vector_params.distance == Distance.COSINE


# ---------------------------------------------------------------------------
# Task 2.3 — Upsert batch
# ---------------------------------------------------------------------------


class TestUpsertBatch:
    def test_upsert_called_with_correct_point_count(self) -> None:
        client = MagicMock()
        records = [_make_record(i) for i in range(5)]
        upsert_batch(client, records)
        client.upsert.assert_called_once()
        _, kwargs = client.upsert.call_args
        assert len(kwargs["points"]) == 5

    def test_upsert_not_called_for_empty_batch(self) -> None:
        client = MagicMock()
        upsert_batch(client, [])
        client.upsert.assert_not_called()

    def test_point_payload_contains_required_keys(self) -> None:
        client = MagicMock()
        upsert_batch(client, [_make_record(42)])
        _, kwargs = client.upsert.call_args
        point = kwargs["points"][0]
        assert point.id == 42
        assert set(point.payload.keys()) >= {"brand", "price", "size", "image_url"}

    def test_upsert_targets_correct_collection(self) -> None:
        client = MagicMock()
        upsert_batch(client, [_make_record()])
        _, kwargs = client.upsert.call_args
        assert kwargs["collection_name"] == COLLECTION_NAME

    def test_vector_converted_to_list(self) -> None:
        client = MagicMock()
        upsert_batch(client, [_make_record()])
        _, kwargs = client.upsert.call_args
        point = kwargs["points"][0]
        assert isinstance(point.vector, list)
        assert len(point.vector) == VECTOR_SIZE


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Task 2.2 — Embedding output shape (real model call, skipped if unavailable)
# ---------------------------------------------------------------------------


class TestEmbeddingOutput:
    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch not importable in this environment",
    )
    def test_embedding_shape_and_dtype(self) -> None:
        from PIL import Image

        from MedClip_pipeline.embeddings.pipeline import embed_image, load_model

        model, preprocess = load_model()
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        vector = embed_image(image, model, preprocess)

        assert vector.shape == (VECTOR_SIZE,), (
            f"Expected ({VECTOR_SIZE},), got {vector.shape}"
        )
        assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch not importable in this environment",
    )
    def test_embedding_is_l2_normalized(self) -> None:
        from PIL import Image

        from MedClip_pipeline.embeddings.pipeline import embed_image, load_model

        model, preprocess = load_model()
        image = Image.new("RGB", (224, 224), color=(128, 64, 200))
        vector = embed_image(image, model, preprocess)

        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-5, f"Vector not unit-normalized, norm={norm}"
