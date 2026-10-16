"""
CLI entry point for the Vinted scraper.

Usage (run from repo root):
    uv run python -m MedClip_pipeline.scraper <pages> [--search <text>]

Examples:
    uv run python -m MedClip_pipeline.scraper 5
    uv run python -m MedClip_pipeline.scraper 10 --search "vintage jacket"
"""

import argparse
import asyncio
import logging

from pydantic import ValidationError
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


async def run(pages: int, search_text: str) -> None:
    await init_db()

    saved = skipped = invalid = 0

    async for db in get_session():
        async with VintedSession() as session:
            async for raw in fetch_catalog(
                session, pages=pages, search_text=search_text
            ):
                try:
                    item = VintedItem.model_validate(raw)
                except ValidationError as exc:
                    logger.debug("Quarantined item %s: %s", raw.get("id"), exc)
                    invalid += 1
                    continue

                # Skip duplicates already in the database.
                existing = await db.get(Product, item.id)
                if existing:
                    skipped += 1
                    continue

                db.add(
                    Product(
                        id=item.id,
                        title=item.title,
                        price=item.price.amount,
                        currency=item.price.currency_code,
                        brand=item.brand_title,
                        size=item.size_title,
                        image_url=item.image_url,
                        category=item.category,
                    )
                )
                saved += 1

        await db.commit()

    logger.info(
        "Scrape complete — saved: %s | skipped (duplicate): %s | invalid: %s",
        saved,
        skipped,
        invalid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Vinted catalog into PostgreSQL"
    )
    parser.add_argument(
        "pages", type=int, help="Number of catalog pages to fetch (96 items/page)"
    )
    parser.add_argument(
        "--search", default="", metavar="TEXT", help="Optional search query"
    )
    args = parser.parse_args()

    asyncio.run(run(pages=args.pages, search_text=args.search))


if __name__ == "__main__":
    main()
