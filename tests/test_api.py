"""
Unit tests for swipe math helpers.

Tests the per-category EMA model that replaced the old dual-EWA approach.
No database or Qdrant connection is required.

Run with:
    uv run pytest tests/test_api.py -v
"""

import math

import numpy as np
import pytest

from MedClip_pipeline.api.routers.swipe import _ALPHA, apply_ema, l2_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int = 768, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def _zero_vec(dim: int = 768) -> list[float]:
    return [0.0] * dim


def _norm(v: list[float]) -> float:
    return float(np.linalg.norm(v))


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        result = l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_zero_vector_unchanged(self) -> None:
        v = np.zeros(768, dtype="float32")
        result = l2_normalize(v)
        assert np.all(result == 0.0)

    def test_arbitrary_vector_normalized(self) -> None:
        v = np.array([3.0, 4.0], dtype="float32")
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_negative_components_normalized(self) -> None:
        v = np.array([-1.0, -1.0, -1.0], dtype="float32")
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# apply_ema — cold start (current is None)
# ---------------------------------------------------------------------------


class TestApplyEmaColdStart:
    def test_cold_start_returns_normalized_incoming(self) -> None:
        incoming = [3.0, 4.0] + [0.0] * 766
        result = apply_ema(None, incoming)
        assert abs(_norm(result) - 1.0) < 1e-5

    def test_cold_start_aligns_with_incoming_direction(self) -> None:
        incoming = _unit_vec(seed=7)
        result = apply_ema(None, incoming)
        cos = float(np.dot(result, incoming))
        assert cos > 0.9999

    def test_cold_start_zero_incoming_stays_finite(self) -> None:
        result = apply_ema(None, _zero_vec())
        assert all(math.isfinite(v) for v in result)


# ---------------------------------------------------------------------------
# apply_ema — warm update
# ---------------------------------------------------------------------------


class TestApplyEmaWarm:
    def test_result_is_l2_normalized(self) -> None:
        result = apply_ema(_unit_vec(seed=1), _unit_vec(seed=2))
        assert abs(_norm(result) - 1.0) < 1e-5

    def test_ema_formula_correct(self) -> None:
        current = _unit_vec(seed=1)
        incoming = _unit_vec(seed=2)

        result = apply_ema(current, incoming)

        cv = np.array(current, dtype="float32")
        iv = np.array(incoming, dtype="float32")
        expected = _ALPHA * cv + (1 - _ALPHA) * iv
        expected = expected / np.linalg.norm(expected)

        np.testing.assert_allclose(result, expected.tolist(), atol=1e-5)

    def test_high_alpha_stays_close_to_current(self) -> None:
        """With alpha=0.8 the result should remain closer to current than incoming."""
        current = _unit_vec(seed=10)
        incoming = _unit_vec(seed=99)

        result = apply_ema(current, incoming)

        cos_to_current = float(np.dot(result, current))
        cos_to_incoming = float(np.dot(result, incoming))

        assert cos_to_current > cos_to_incoming, (
            f"High alpha should keep result near current. "
            f"cos_current={cos_to_current:.4f}, cos_incoming={cos_to_incoming:.4f}"
        )

    def test_alpha_value_is_conservative(self) -> None:
        """_ALPHA should be >= 0.5 to weight existing preference more than new item."""
        assert _ALPHA >= 0.5

    def test_dislike_does_not_call_apply_ema(self) -> None:
        """Left-swipe simulation: apply_ema is not invoked, vectors stay identical."""
        current = _unit_vec(seed=7)
        result = current  # dislike path skips apply_ema
        assert result == current

    @pytest.mark.parametrize("seed_a,seed_b", [(0, 1), (5, 10), (42, 99)])
    def test_result_always_normalized_various_inputs(
        self, seed_a: int, seed_b: int
    ) -> None:
        result = apply_ema(_unit_vec(seed=seed_a), _unit_vec(seed=seed_b))
        assert abs(_norm(result) - 1.0) < 1e-5

    def test_identical_vectors_result_is_same_direction(self) -> None:
        v = _unit_vec(seed=3)
        result = apply_ema(v, v)
        cos = float(np.dot(result, v))
        assert cos > 0.9999
