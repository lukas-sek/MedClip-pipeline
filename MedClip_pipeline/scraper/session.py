import asyncio
import logging
import os
from functools import wraps
from typing import Any, Callable

from curl_cffi.requests import AsyncSession, Response
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL: str = os.getenv("BASE_URL", "https://www.vinted.de")

_RETRY_SLEEP_SECONDS = 10


def _self_healing(method: Callable) -> Callable:
    """Decorator: on 401/403, re-handshake and retry the request once."""

    @wraps(method)
    async def wrapper(self: "VintedSession", *args: Any, **kwargs: Any) -> Response:
        response: Response = await method(self, *args, **kwargs)
        if response.status_code in (401, 403):
            logger.warning(
                "Received %s — sleeping %ss then re-handshaking.",
                response.status_code,
                _RETRY_SLEEP_SECONDS,
            )
            await asyncio.sleep(_RETRY_SLEEP_SECONDS)
            await self.handshake()
            response = await method(self, *args, **kwargs)
        return response

    return wrapper


class VintedSession:
    """
    Async HTTP session that impersonates a desktop browser and maintains
    a valid Vinted access token. All requests are automatically retried
    once after a fresh handshake on 401/403.
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session: AsyncSession = AsyncSession(impersonate="chrome124")
        self._token: str | None = None

    async def handshake(self) -> None:
        """
        Fire an initial request to the root domain to establish a valid
        session and extract the public access_token_web.
        """
        logger.info("Performing handshake with %s", self.base_url)
        response = await self._session.get(self.base_url)
        response.raise_for_status()

        # Token is typically present in cookies after the initial page load.
        token = self._session.cookies.get("access_token_web")
        if token:
            self._token = token
            logger.info("access_token_web extracted from cookies.")
            return

        # Fallback: some locales expose it in a JSON response or header.
        try:
            data = response.json()
            self._token = data.get("access_token_web") or data.get("token")
        except Exception:
            pass

        if self._token:
            logger.info("access_token_web extracted from response body.")
        else:
            logger.warning(
                "access_token_web not found — proceeding with session cookies only."
            )

    @_self_healing
    async def get(self, url: str, **kwargs: Any) -> Response:
        """GET with self-healing on auth errors."""
        if self._token:
            headers = kwargs.pop("headers", {})
            headers.setdefault("Authorization", f"Bearer {self._token}")
            kwargs["headers"] = headers
        return await self._session.get(url, **kwargs)

    async def __aenter__(self) -> "VintedSession":
        await self.handshake()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._session.close()
