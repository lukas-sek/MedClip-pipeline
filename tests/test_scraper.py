"""
Phase 1 tests — Pydantic validation and catalog fetch logic.

Run with:
    uv run pytest tests/test_scraper.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from MedClip_pipeline.scraper.models import VintedItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_item(**overrides: object) -> dict:
    """Return a minimal valid raw Vinted API item dict."""
    base: dict = {
        "id": 123,
        "title": "Vintage Levi's Jacket",
        "price": {"amount": "29.99", "currency_code": "EUR"},
        "brand_title": "Levi's",
        "size_title": "M",
        "photos": [{"full_size_url": "https://images.vinted.net/item.jpg"}],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task 1.3 — Pydantic schema validation
# ---------------------------------------------------------------------------


class TestVintedItemValidation:
    def test_valid_item_is_accepted(self) -> None:
        item = VintedItem.model_validate(_make_raw_item())
        assert item.id == 123
        assert item.image_url == "https://images.vinted.net/item.jpg"
        assert item.price.amount == "29.99"
        assert item.price.currency_code == "EUR"

    def test_missing_photos_is_quarantined(self) -> None:
        raw = _make_raw_item(photos=[])
        with pytest.raises(Exception, match="no usable image_url"):
            VintedItem.model_validate(raw)

    def test_none_photos_is_quarantined(self) -> None:
        raw = _make_raw_item(photos=None)
        with pytest.raises(Exception, match="no usable image_url"):
            VintedItem.model_validate(raw)

    def test_missing_price_is_quarantined(self) -> None:
        raw = _make_raw_item(price=None)
        with pytest.raises(Exception, match="no price object"):
            VintedItem.model_validate(raw)

    def test_non_numeric_price_amount_is_quarantined(self) -> None:
        raw = _make_raw_item(price={"amount": "free", "currency_code": "EUR"})
        with pytest.raises(Exception, match="not a valid decimal"):
            VintedItem.model_validate(raw)

    def test_optional_brand_and_size_are_none(self) -> None:
        raw = _make_raw_item()
        raw.pop("brand_title", None)
        raw.pop("size_title", None)
        item = VintedItem.model_validate(raw)
        assert item.brand_title is None
        assert item.size_title is None

    def test_photo_url_fallback_key(self) -> None:
        """Falls back to 'url' key if 'full_size_url' is absent."""
        raw = _make_raw_item(photos=[{"url": "https://images.vinted.net/fallback.jpg"}])
        item = VintedItem.model_validate(raw)
        assert item.image_url == "https://images.vinted.net/fallback.jpg"


# ---------------------------------------------------------------------------
# Task 1.2 — Catalog fetch (mocked session)
# ---------------------------------------------------------------------------


class TestFetchCatalog:
    @pytest.mark.asyncio
    async def test_yields_items_from_single_page(self) -> None:
        from MedClip_pipeline.scraper.catalog import fetch_catalog

        raw_items = [_make_raw_item(id=i) for i in range(1, 4)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": raw_items}

        mock_session = MagicMock()
        mock_session.base_url = "https://www.vinted.fr"
        mock_session.get = AsyncMock(return_value=mock_response)

        collected = [item async for item in fetch_catalog(mock_session, pages=1)]
        assert len(collected) == 3
        assert collected[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_empty_items_list_yields_nothing(self) -> None:
        from MedClip_pipeline.scraper.catalog import fetch_catalog

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}

        mock_session = MagicMock()
        mock_session.base_url = "https://www.vinted.fr"
        mock_session.get = AsyncMock(return_value=mock_response)

        collected = [item async for item in fetch_catalog(mock_session, pages=1)]
        assert collected == []

    @pytest.mark.asyncio
    async def test_rate_limit_delay_is_applied_between_pages(self) -> None:
        from MedClip_pipeline.scraper.catalog import fetch_catalog

        raw_items = [_make_raw_item(id=1)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": raw_items}

        mock_session = MagicMock()
        mock_session.base_url = "https://www.vinted.fr"
        mock_session.get = AsyncMock(return_value=mock_response)

        with patch(
            "MedClip_pipeline.scraper.catalog.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            _ = [item async for item in fetch_catalog(mock_session, pages=3)]
            # delay is called between pages: pages-1 times
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_non_200_response_yields_nothing(self) -> None:
        from MedClip_pipeline.scraper.catalog import fetch_catalog

        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_session = MagicMock()
        mock_session.base_url = "https://www.vinted.fr"
        mock_session.get = AsyncMock(return_value=mock_response)

        collected = [item async for item in fetch_catalog(mock_session, pages=1)]
        assert collected == []
