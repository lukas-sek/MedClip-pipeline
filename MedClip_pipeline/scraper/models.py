from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

# Maps keywords found in Vinted catalog/category titles (and item titles as
# fallback) to the standard category names used by the feed/vector system.
# Checked in order — first match wins.
_KEYWORD_MAP: list[tuple[str, str]] = [
    # jackets / outerwear — EN / FR / DE
    ("jacket", "jackets"),
    ("veste", "jackets"),
    ("manteau", "jackets"),
    ("coat", "jackets"),
    ("blouson", "jackets"),
    ("parka", "jackets"),
    ("hoodie", "jackets"),
    ("sweatshirt", "jackets"),
    ("jacke", "jackets"),
    ("mantel", "jackets"),
    ("anorak", "jackets"),
    ("windbreaker", "jackets"),
    ("übergangsjacke", "jackets"),
    # pants — EN / FR / DE
    ("pant", "pants"),
    ("jean", "pants"),
    ("pantalon", "pants"),
    ("trouser", "pants"),
    ("legging", "pants"),
    ("short", "pants"),
    ("hose", "pants"),
    ("jeans", "pants"),
    ("chino", "pants"),
    ("shorts", "pants"),
    # shirts / tops — EN / FR / DE
    ("shirt", "shirts"),
    ("chemise", "shirts"),
    ("t-shirt", "shirts"),
    ("tshirt", "shirts"),
    ("top", "shirts"),
    ("blouse", "shirts"),
    ("polo", "shirts"),
    ("hemd", "shirts"),
    ("bluse", "shirts"),
    ("oberteil", "shirts"),
    ("pullover", "shirts"),
    ("pulli", "shirts"),
    ("strickpullover", "shirts"),
    # shoes — EN / FR / DE
    ("shoe", "shoes"),
    ("chaussure", "shoes"),
    ("sneaker", "shoes"),
    ("boot", "shoes"),
    ("botte", "shoes"),
    ("basket", "shoes"),
    ("loafer", "shoes"),
    ("sandal", "shoes"),
    ("schuh", "shoes"),
    ("stiefel", "shoes"),
    ("turnschuh", "shoes"),
    ("sandale", "shoes"),
    ("pumps", "shoes"),
    ("slipper", "shoes"),
    # hats — EN / FR / DE
    ("hat", "hats"),
    ("cap", "hats"),
    ("chapeau", "hats"),
    ("bonnet", "hats"),
    ("casquette", "hats"),
    ("mütze", "hats"),
    ("kappe", "hats"),
    ("hut", "hats"),
    ("beanie", "hats"),
    # accessories — EN / FR / DE
    ("bag", "accessories"),
    ("sac", "accessories"),
    ("belt", "accessories"),
    ("ceinture", "accessories"),
    ("scarf", "accessories"),
    ("écharpe", "accessories"),
    ("watch", "accessories"),
    ("montre", "accessories"),
    ("jewel", "accessories"),
    ("bijou", "accessories"),
    ("wallet", "accessories"),
    ("tasche", "accessories"),
    ("gürtel", "accessories"),
    ("schal", "accessories"),
    ("uhr", "accessories"),
    ("schmuck", "accessories"),
    ("portemonnaie", "accessories"),
    ("geldbörse", "accessories"),
    ("rucksack", "accessories"),
    ("handtasche", "accessories"),
    ("sonnenbrille", "accessories"),
]


def _classify(text: str) -> str:
    """Return the first matching standard category or 'unknown'."""
    lower = text.lower()
    for keyword, category in _KEYWORD_MAP:
        if keyword in lower:
            return category
    return "unknown"


def _extract_category(data: dict) -> str:
    """
    Try Vinted's catalog/category fields first, fall back to title keywords.
    Vinted items may carry: catalog.title, category.title, catalog_title, etc.
    """
    for key in ("catalog", "category"):
        obj = data.get(key)
        if isinstance(obj, dict):
            title = obj.get("title") or obj.get("name") or ""
            if title:
                cat = _classify(title)
                if cat != "unknown":
                    return cat

    for key in ("catalog_title", "category_title"):
        title = data.get(key) or ""
        if title:
            cat = _classify(str(title))
            if cat != "unknown":
                return cat

    # Last resort: keyword-match the item title itself.
    return _classify(data.get("title", ""))


class VintedPrice(BaseModel):
    amount: str
    currency_code: str

    @field_validator("amount")
    @classmethod
    def amount_must_be_numeric(cls, v: str) -> str:
        try:
            Decimal(v)
        except InvalidOperation:
            raise ValueError(f"price amount is not a valid decimal: {v!r}")
        return v


class VintedItem(BaseModel):
    id: int
    title: str
    price: VintedPrice
    brand_title: str | None = None
    size_title: str | None = None
    image_url: str
    category: str = "unknown"

    @model_validator(mode="before")
    @classmethod
    def extract_fields(cls, data: Any) -> Any:
        """
        Map raw Vinted API JSON onto the flat schema expected by this model.
        Raises ValueError (quarantine) if image or price data is missing/malformed.
        """
        if not isinstance(data, dict):
            return data

        # Extract image URL from nested photos list.
        photos = data.get("photos") or data.get("photo") or []
        if isinstance(photos, dict):
            photos = [photos]
        url: str | None = None
        for photo in photos:
            url = photo.get("full_size_url") or photo.get("url")
            if url:
                break
        if not url:
            raise ValueError("item has no usable image_url — quarantined")

        # Map price object: Vinted returns {"amount": "12.00", "currency_code": "EUR"}
        raw_price = data.get("price")
        if not raw_price or not isinstance(raw_price, dict):
            raise ValueError("item has no price object — quarantined")

        return {
            "id": data["id"],
            "title": data.get("title", ""),
            "price": raw_price,
            "brand_title": data.get("brand_title"),
            "size_title": data.get("size_title"),
            "image_url": url,
            "category": _extract_category(data),
        }
