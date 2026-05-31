"""I/O helpers: loading images, saving PNGs, loading/saving JSON metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image_color(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8, raise if not found or unreadable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    logger.debug("Loaded %s  shape=%s", path.name, img.shape)
    return img


def load_image_gray(path: str | Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_png(img: np.ndarray, path: str | Path, *, overwrite: bool = False) -> Path:
    """Save a uint8 or float64 array as a PNG.

    Float arrays are clipped to [0,255] and converted to uint8.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use --overwrite): {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {path}")
    logger.debug("Saved %s", path)
    return path


def save_json(data: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    """Serialize *data* to a pretty-printed JSON file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.debug("Saved JSON %s", path)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def bgr_to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → float64 luminance in [0, 255]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64)


def gray_uint8_to_float(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale → float64 in [0, 255]."""
    return img.astype(np.float64)


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """float64 [0,255] → uint8 with clipping."""
    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_output_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
