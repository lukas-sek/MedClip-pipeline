"""Load sample frames from a directory for the MedClip pipeline.

Intended for layouts that mimic Cholec80 surgical video frames or
histopathology tiles (flat folders or class/video subdirectories of images).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    image: Image.Image


def preprocess_frame(frame: np.ndarray) -> Image.Image:
    """Crop border pixels, center-crop to a square, and return RGB PIL data."""
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width = frame.shape[:2]
    border_y, border_x = max(1, height // 50), max(1, width // 50)
    frame = frame[border_y : height - border_y, border_x : width - border_x]
    height, width = frame.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    frame = frame[top : top + side, left : left + side]
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return Image.fromarray(frame).convert("RGB")


def load_samples(input_dir: str | Path, limit: int | None = None) -> list[ImageSample]:
    """Load image frames or histopathology tiles from a directory."""
    paths = sorted(
        path
        for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]

    samples: list[ImageSample] = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is not None:
            samples.append(ImageSample(path=path, image=preprocess_frame(frame)))
    return samples


def summarize(input_dir: Path, limit: int | None) -> int:
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        return 1

    samples = load_samples(input_dir, limit=limit)
    if not samples:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        print(
            f"No supported images found in '{input_dir}'. "
            f"Expected: {extensions}",
            file=sys.stderr,
        )
        return 1

    by_class = Counter(sample.path.parent.name for sample in samples)
    print(f"Loaded {len(samples)} frame(s) from {input_dir.resolve()}")
    print(f"Classes / parent folders: {len(by_class)}")
    for name, count in sorted(by_class.items()):
        print(f"  {name}: {count}")
    print()
    print("Samples:")
    for sample in samples:
        width, height = sample.image.size
        print(f"  {sample.path}  ({width}x{height}, mode={sample.image.mode})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load sample frames from an input directory "
            "(e.g. Cholec80-style surgical frames or histopathology tiles)."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing image frames or tiles",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of frames to load",
    )
    args = parser.parse_args(argv)
    return summarize(args.input_dir, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
