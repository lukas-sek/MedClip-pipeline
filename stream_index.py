"""Stream a Hugging Face dataset and index BiomedCLIP embeddings into Qdrant.

No local zip downloads required — images are pulled via streaming.
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from PIL import Image

from database import get_client, recreate_collection, upsert_embeddings
from preprocess import embed_images, load_model

DEFAULT_DATASET = "marmal88/skin_cancer"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 50
DEFAULT_BATCH_SIZE = 16
CACHE_DIR = Path("data/cache")


def _as_rgb(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image.convert("RGB")


def _cache_image(image: Image.Image, image_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{image_id}.jpg"
    image.save(path, format="JPEG", quality=90)
    return path


def _payload(item: dict[str, Any], dataset_name: str, split: str, index: int) -> dict:
    image_id = str(item.get("image_id") or f"{index:05d}")
    label = item.get("dx") or item.get("label")
    return {
        "image_path": f"hf://{dataset_name}/{split}/{image_id}",
        "filename": image_id,
        "media_url": f"/media/{image_id}.jpg",
        "modality": "Dermatology",
        "source": f"HuggingFace-{dataset_name}",
        "class": str(label) if label is not None else None,
        "dx_type": item.get("dx_type"),
        "localization": item.get("localization"),
        "timestamp": 0.0,
    }


def _sample_stream(
    dataset_name: str,
    split: str,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Reservoir-sample from a HF stream (avoids Arrow shuffle offset overflow)."""
    # Scan more rows than we keep so labels aren't one contiguous class block.
    scan = max(limit * 20, 1_000)
    rng = random.Random(seed)
    stream: Iterator[dict[str, Any]] = load_dataset(
        dataset_name, split=split, streaming=True
    )
    reservoir: list[dict[str, Any]] = []
    for i, item in enumerate(stream):
        if i >= scan:
            break
        if len(reservoir) < limit:
            reservoir.append(item)
            continue
        j = rng.randint(0, i)
        if j < limit:
            reservoir[j] = item
    rng.shuffle(reservoir)
    return reservoir[:limit]


async def stream_and_index(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    limit: int = DEFAULT_LIMIT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = True,
) -> int:
    """Stream dataset rows from Hugging Face, embed, and upsert into Qdrant."""
    processor, model, device = load_model()
    print(f"Sampling up to {limit} images from {dataset_name}/{split} ...")
    samples = _sample_stream(dataset_name, split, limit)

    client = get_client()
    indexed = 0
    try:
        if replace:
            print(f"Recreating Qdrant collection so only cached images remain ...")
            await recreate_collection(client)
        else:
            from database import ensure_collection

            await ensure_collection(client)

        images: list[Image.Image] = []
        metadata: list[dict] = []

        for index, item in enumerate(samples):
            image = _as_rgb(item["image"])
            payload = _payload(item, dataset_name, split, index)
            _cache_image(image, payload["filename"])
            images.append(image)
            metadata.append(payload)

            if len(images) >= batch_size:
                embeddings = embed_images(images, processor, model, device)
                await upsert_embeddings(client, embeddings, metadata)
                indexed += len(metadata)
                print(f"Indexed {indexed}/{len(samples)} ...")
                images.clear()
                metadata.clear()

        if images:
            embeddings = embed_images(images, processor, model, device)
            await upsert_embeddings(client, embeddings, metadata)
            indexed += len(metadata)
    finally:
        await client.close()

    print(
        f"Successfully streamed and indexed {indexed} items "
        f"from Hugging Face ({dataset_name}/{split})."
    )
    return indexed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Hugging Face image dataset and index BiomedCLIP "
            "embeddings into Qdrant (no local zips)."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of streamed samples to index",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not recreate the collection (may leave uncached stale points)",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        stream_and_index(
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            replace=not args.keep_existing,
        )
    )


if __name__ == "__main__":
    main()
