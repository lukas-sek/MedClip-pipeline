"""
Phase 4 MLOps tests — deterministic vector integrity assertions.

These tests validate the mathematical guarantees required for correct
Qdrant ANN behaviour:
  1. Right swipe moves the user vector strictly closer to the liked item.
  2. The updated vector is always L2-normalized (norm == 1.0).
  3. The fashion encoder produces a (768,) float32 output with no NaN values.

Run with:
    uv run pytest tests/test_mlops.py -v
"""

import numpy as np
import pytest

from MedClip_pipeline.api.routers.swipe import update_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(seed: int, dim: int = 768) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype="float32")
    vb = np.array(b, dtype="float32")
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — Vector Shift Assertion
# ---------------------------------------------------------------------------


class TestVectorShift:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_right_swipe_moves_user_closer_to_item(self, seed: int) -> None:
        """
        After a right swipe, cosine similarity between U_t+1 and I must be
        strictly greater than the similarity between U_t and I.
        """
        u_long_t = _unit_vec(seed=seed)
        u_short_t = _unit_vec(seed=seed + 100)
        item_vec = _unit_vec(seed=seed + 200)

        sim_before = _cosine_similarity(u_long_t, item_vec)

        new_long, _ = update_vectors(u_long_t, u_short_t, item_vec)

        sim_after = _cosine_similarity(new_long, item_vec)

        assert sim_after > sim_before or np.isclose(sim_after, 1.0, atol=1e-5), (
            f"seed={seed}: expected sim_after ({sim_after:.6f}) > "
            f"sim_before ({sim_before:.6f})"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_short_term_also_moves_closer_to_item(self, seed: int) -> None:
        u_long_t = _unit_vec(seed=seed)
        u_short_t = _unit_vec(seed=seed + 100)
        item_vec = _unit_vec(seed=seed + 200)

        sim_before = _cosine_similarity(u_short_t, item_vec)
        _, new_short = update_vectors(u_long_t, u_short_t, item_vec)
        sim_after = _cosine_similarity(new_short, item_vec)

        assert sim_after > sim_before or np.isclose(sim_after, 1.0, atol=1e-5), (
            f"seed={seed}: short-term sim_after ({sim_after:.6f}) > "
            f"sim_before ({sim_before:.6f})"
        )


# ---------------------------------------------------------------------------
# Test 2 — Normalization Check
# ---------------------------------------------------------------------------


class TestNormalization:
    @pytest.mark.parametrize("seed", [0, 5, 99])
    def test_long_term_vector_norm_is_one(self, seed: int) -> None:
        """
        U_t+1 must always have norm ≈ 1.0.
        Un-normalized vectors corrupt Qdrant cosine distance calculations.
        """
        new_long, _ = update_vectors(
            _unit_vec(seed=seed),
            _unit_vec(seed=seed + 1),
            _unit_vec(seed=seed + 2),
        )
        norm = float(np.linalg.norm(new_long))
        assert np.isclose(norm, 1.0, atol=1e-5), (
            f"seed={seed}: long_term norm={norm:.8f}, expected 1.0"
        )

    @pytest.mark.parametrize("seed", [0, 5, 99])
    def test_short_term_vector_norm_is_one(self, seed: int) -> None:
        _, new_short = update_vectors(
            _unit_vec(seed=seed),
            _unit_vec(seed=seed + 1),
            _unit_vec(seed=seed + 2),
        )
        norm = float(np.linalg.norm(new_short))
        assert np.isclose(norm, 1.0, atol=1e-5), (
            f"seed={seed}: short_term norm={norm:.8f}, expected 1.0"
        )

    def test_normalization_holds_after_zero_user_vector(self) -> None:
        """Edge case: user vector starts at zero (new user)."""
        zero = [0.0] * 768
        item_vec = _unit_vec(seed=7)
        new_long, new_short = update_vectors(zero, zero, item_vec)
        assert np.isclose(np.linalg.norm(new_long), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(new_short), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3 — Model Inference Output
# ---------------------------------------------------------------------------


class TestModelInference:
    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch not available in this environment",
    )
    def test_encoder_output_shape_is_768(self) -> None:
        """
        Pass a dummy [224, 224] PIL image through the marqo-fashionSigLIP
        encoder. Assert output shape is (768,) and dtype is float32.
        """
        from PIL import Image

        from MedClip_pipeline.embeddings.pipeline import embed_image, load_model

        model, preprocess = load_model()
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        vector = embed_image(image, model, preprocess)

        assert vector.shape == (768,), f"Expected (768,), got {vector.shape}"
        assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch not available in this environment",
    )
    def test_encoder_output_contains_no_nan(self) -> None:
        """
        NaN values in the embedding vector would silently corrupt all
        downstream Qdrant ANN results.
        """
        from PIL import Image

        from MedClip_pipeline.embeddings.pipeline import embed_image, load_model

        model, preprocess = load_model()
        image = Image.new("RGB", (224, 224), color=(255, 0, 128))
        vector = embed_image(image, model, preprocess)

        assert not np.any(np.isnan(vector)), "Encoder output contains NaN values"
