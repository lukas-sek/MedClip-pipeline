"""
Phase 1 entry point.

Usage:
    uv run python -m MedClip_pipeline.pipeline --pages 3 --search "vintage jacket"

Environment variables (see .env.example):
    DATABASE_URL  — async PostgreSQL DSN  (required)
    BASE_URL      — Vinted locale root    (default: https://www.vinted.fr)
"""

import argparse
import asyncio
import logging
from decimal import Decimal

from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession

from MedClip_pipeline.db.engine import get_session, init_db
from MedClip_pipeline.db.models import Product
from MedClip_pipeline.scraper.catalog import fetch_catalog
from MedClip_pipeline.scraper.models import VintedItem
from MedClip_pipeline.scraper.session import VintedSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def _upsert_product(session: AsyncSession, item: VintedItem) -> None:
    """Insert or update a single validated product row."""
    product = Product(
        id=item.id,
        title=item.title,
        price=Decimal(item.price.amount),
        currency=item.price.currency_code,
        brand=item.brand_title,
        size=item.size_title,
        image_url=item.image_url,
    )
    await session.merge(product)


async def run(pages: int, search_text: str) -> None:
    await init_db()

    valid_count = 0
    skipped_count = 0

    async with VintedSession() as vinted:
        async for db_session in get_session():
            async for raw_item in fetch_catalog(
                vinted, pages=pages, search_text=search_text
            ):
                try:
                    item = VintedItem.model_validate(raw_item)
                except ValidationError as exc:
                    logger.warning(
                        "Quarantined item id=%s: %s",
                        raw_item.get("id", "?"),
                        exc.errors()[0]["msg"],
                    )
                    skipped_count += 1
                    continue

                await _upsert_product(db_session, item)
                valid_count += 1

            await db_session.commit()

    logger.info(
        "Pipeline complete — stored: %s | quarantined: %s",
        valid_count,
        skipped_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MedClip-pipeline Vinted scraper pipeline")
    parser.add_argument(
        "--pages", type=int, default=1, help="Number of catalog pages to fetch"
    )
    parser.add_argument("--search", type=str, default="", help="Search text filter")
    args = parser.parse_args()

    asyncio.run(run(pages=args.pages, search_text=args.search))


if __name__ == "__main__":
    main()
