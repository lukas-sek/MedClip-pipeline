from pathlib import Path

import numpy as np
from PIL import Image

from load_frames import IMAGE_SIZE, load_samples, preprocess_frame


def test_preprocess_frame_returns_square_rgb() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    image = preprocess_frame(frame)
    assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert image.mode == "RGB"


def test_load_samples_from_directory(tmp_path: Path) -> None:
    path = tmp_path / "tile.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(path)

    samples = load_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].path == path
    assert samples[0].image.size == (IMAGE_SIZE, IMAGE_SIZE)
