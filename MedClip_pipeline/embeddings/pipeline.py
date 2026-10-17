"""
Phase 2 embedding pipeline — offline batch script.

Pulls unembedded Product rows from PostgreSQL, downloads each image,
runs it through Marqo/marqo-fashionSigLIP on CPU, and upserts the
resulting 768-dim vectors into Qdrant.

Usage:
    uv run python -m MedClip_pipeline.embeddings.pipeline --batch-size 32

Environment variables (see .env.example):
    DATABASE_URL     — async PostgreSQL DSN  (required)
    QDRANT_HOST      — Qdrant hostname       (default: localhost)
    QDRANT_GRPC_PORT — Qdrant gRPC port      (default: 6334)
"""

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from curl_cffi.requests import AsyncSession
from PIL import Image
from sqlalchemy import select

from MedClip_pipeline.db.engine import get_session, init_db
from MedClip_pipeline.db.models import Product
from MedClip_pipeline.embeddings.vector_store import (
    EmbedRecord,
    get_client,
    init_collection,
    upsert_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# open_clip hub reference — avoids the custom transformers wrapper
# which has meta-tensor incompatibilities with newer PyTorch.
MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"

# Type aliases for the open_clip objects returned by load_model.
OpenClipModel = torch.nn.Module
OpenClipPreprocess = object  # torchvision transform callable

# Text prompts used for zero-shot category classification.
# Keys must match the category values stored in Product and Qdrant payloads.
CATEGORY_PROMPTS: dict[str, str] = {
    "jackets": "a photo of a jacket, coat, or outerwear garment",
    "pants": "a photo of pants, jeans, trousers, or shorts",
    "shirts": "a photo of a shirt, t-shirt, top, blouse, or sweater",
    "shoes": "a photo of shoes, boots, sneakers, or sandals",
    "hats": "a photo of a hat, cap, or beanie",
    "accessories": "a photo of a fashion accessory: bag, belt, scarf, or wallet",
}


def load_model() -> tuple[OpenClipModel, OpenClipPreprocess]:
    logger.info("Loading model '%s' on CPU…", MODEL_ID)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval()
    logger.info("Model loaded.")
    return model, preprocess


def build_category_classifier(
    model: OpenClipModel,
) -> tuple[list[str], np.ndarray]:
    """
    Encode the category text prompts with the model's text tower.
    Returns (category_names, centroids) where centroids has shape (N, 768).
    Called once after load_model() — cost is negligible compared to image encoding.
    """
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    names = list(CATEGORY_PROMPTS.keys())
    texts = tokenizer(list(CATEGORY_PROMPTS.values()))
    with torch.no_grad():
        text_feats = model.encode_text(texts)  # type: ignore[attr-defined]
    centroids = F.normalize(text_feats, p=2, dim=-1).cpu().numpy().astype("float32")
    logger.info("Built category classifier for %s categories.", len(names))
    return names, centroids


def classify_embedding(
    vector: np.ndarray,
    category_names: list[str],
    centroids: np.ndarray,
) -> str:
    """Return the category whose text embedding is most similar to vector."""
    sims = centroids @ vector  # shape (N,)
    return category_names[int(np.argmax(sims))]


def embed_image(
    image: Image.Image,
    model: OpenClipModel,
    preprocess: OpenClipPreprocess,
) -> np.ndarray:
    """
    Run a single PIL image through the vision encoder.
    Returns a normalized float32 array of shape (768,).
    """
    tensor = preprocess(image).unsqueeze(0)  # type: ignore[operator]
    with torch.no_grad():
        features = model.encode_image(tensor)
    vector = F.normalize(features, p=2, dim=-1)
    return vector.cpu().numpy().astype("float32").squeeze(0)


async def _download_image(http: AsyncSession, url: str) -> Image.Image | None:
    """Download an image URL and return a PIL Image, or None on failure."""
    try:
        response = await http.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return None


async def run(batch_size: int) -> None:
    await init_db()

    model, preprocess = load_model()
    category_names, centroids = build_category_classifier(model)
    qdrant = get_client()
    init_collection(qdrant)

    total_embedded = 0
    total_failed = 0

    async with AsyncSession(impersonate="chrome124") as http:
        async for db in get_session():
            # Fetch all products not yet embedded.
            result = await db.exec(select(Product).where(Product.embedded_at.is_(None)))
            products: list[Product] = result.scalars().all()
            logger.info("Found %s unembedded products.", len(products))

            batch: list[EmbedRecord] = []
            batch_products: list[Product] = []

            for product in products:
                image = await _download_image(http, product.image_url)
                if image is None:
                    total_failed += 1
                    continue

                vector = embed_image(image, model, preprocess)
                category = classify_embedding(vector, category_names, centroids)
                product.category = category
                batch.append(
                    EmbedRecord(
                        product_id=product.id,
                        vector=vector,
                        brand=product.brand,
                        price=str(product.price),
                        size=product.size,
                        image_url=product.image_url,
                        category=category,
                    )
                )
                batch_products.append(product)

                if len(batch) >= batch_size:
                    upsert_batch(qdrant, batch)
                    now = datetime.now(timezone.utc)
                    for p in batch_products:
                        p.embedded_at = now
                        db.add(p)
                    await db.commit()
                    total_embedded += len(batch)
                    logger.info("Committed batch of %s.", len(batch))
                    batch = []
                    batch_products = []

            # Flush remaining partial batch.
            if batch:
                upsert_batch(qdrant, batch)
                now = datetime.now(timezone.utc)
                for p in batch_products:
                    p.embedded_at = now
                    db.add(p)
                await db.commit()
                total_embedded += len(batch)

    logger.info(
        "Embedding pipeline complete — embedded: %s | failed: %s",
        total_embedded,
        total_failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedClip-pipeline Phase 2 embedding pipeline"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of vectors to upsert per Qdrant batch (default: 32)",
    )
    args = parser.parse_args()
    asyncio.run(run(batch_size=args.batch_size))


if __name__ == "__main__":
    main()
