from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

from load_frames import load_samples

MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OPENCLIP_MODEL_ID = f"hf-hub:{MODEL_ID}"
EMBEDDING_DIM = 512


@dataclass(frozen=True)
class BiomedCLIPProcessor:
    image_transform: Any
    tokenizer: Any


def load_model() -> tuple[BiomedCLIPProcessor, Any, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_transform = open_clip.create_model_from_pretrained(OPENCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
    processor = BiomedCLIPProcessor(image_transform, tokenizer)
    model = model.to(device).eval()
    return processor, model, device


def _normalize(features: torch.Tensor) -> np.ndarray:
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.detach().cpu().numpy().astype(np.float32)


def embed_images(
    images: list[Image.Image],
    processor: BiomedCLIPProcessor,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    if not images:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    inputs = torch.stack([processor.image_transform(image) for image in images]).to(
        device
    )
    with torch.inference_mode():
        features = model.encode_image(inputs)
    embeddings = _normalize(features)
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIM} dimensions, got {embeddings.shape[1]}"
        )
    return embeddings


def embed_directory(
    input_dir: str | Path,
    processor: BiomedCLIPProcessor,
    model: Any,
    device: torch.device,
    limit: int | None = None,
) -> tuple[np.ndarray, list[dict[str, str | float]]]:
    samples = load_samples(input_dir, limit=limit)
    embeddings = embed_images(
        [sample.image for sample in samples], processor, model, device
    )
    metadata = [
        {
            "image_path": str(sample.path),
            "filename": sample.path.name,
            "timestamp": sample.path.stat().st_mtime,
            "class": sample.path.parent.name,
        }
        for sample in samples
    ]
    return embeddings, metadata


def embed_text(
    text: str,
    processor: BiomedCLIPProcessor,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    inputs = processor.tokenizer([text]).to(device)
    with torch.inference_mode():
        features = model.encode_text(inputs)
    embedding = _normalize(features)[0]
    if embedding.shape != (EMBEDDING_DIM,):
        raise RuntimeError(
            f"Expected {EMBEDDING_DIM} dimensions, got {embedding.shape[0]}"
        )
    return embedding
