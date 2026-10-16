import asyncio
import logging
import os
import random
from typing import AsyncGenerator

from .session import VintedSession

logger = logging.getLogger(__name__)

BASE_URL: str = os.getenv("BASE_URL", "https://www.vinted.de")

_CATALOG_ENDPOINT = "/api/v2/catalog/items"
_MIN_DELAY = 3.0
_MAX_DELAY = 7.0


async def fetch_page(
    session: VintedSession,
    page: int,
    search_text: str = "",
    per_page: int = 96,
) -> list[dict]:
    """
    Fetch a single catalog page from the Vinted API.
    Returns a list of raw item dicts, or an empty list on error.
    """
    url = f"{session.base_url}{_CATALOG_ENDPOINT}"
    params: dict = {"page": page, "per_page": per_page}
    if search_text:
        params["search_text"] = search_text

    response = await session.get(url, params=params)

    if not response.status_code == 200:
        logger.error(
            "fetch_page: unexpected status %s for page %s", response.status_code, page
        )
        return []

    try:
        data = response.json()
    except Exception as exc:
        logger.error("fetch_page: failed to parse JSON for page %s: %s", page, exc)
        return []

    items: list[dict] = data.get("items", [])
    logger.info("fetch_page: page=%s returned %s items", page, len(items))
    return items


async def fetch_catalog(
    session: VintedSession,
    pages: int = 1,
    search_text: str = "",
    per_page: int = 96,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields raw item dicts across `pages` catalog pages.
    Enforces a random delay between page fetches to avoid firewall bans.
    """
    for page in range(1, pages + 1):
        items = await fetch_page(
            session, page, search_text=search_text, per_page=per_page
        )
        for item in items:
            yield item

        if page < pages:
            delay = random.uniform(_MIN_DELAY, _MAX_DELAY)
            logger.debug(
                "Rate-limit delay: sleeping %.2fs before page %s", delay, page + 1
            )
            await asyncio.sleep(delay)
