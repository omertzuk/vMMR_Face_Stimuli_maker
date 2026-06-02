"""Oval face mask generation and application.

The SAME boolean mask is used for self / other / morph throughout the pipeline,
ensuring that the SHINE background-zeroing operation is identical across all
images (so relative equalization is unaffected by the mask edge).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]
FloatImg = NDArray[np.float64]


def create_face_mask(
    size: tuple[int, int] = (405, 425),
    height_fraction: float = 0.95,
    width_fraction: float = 0.70,
) -> BoolMask:
    """Create a centred boolean oval mask.

    Parameters
    ----------
    size:
        (height, width) of the mask in pixels — same as the image shape.
    height_fraction:
        Semi-axis in the vertical direction as a fraction of (height / 2).
    width_fraction:
        Semi-axis in the horizontal direction as a fraction of (width / 2).

    Returns
    -------
    mask : bool array of shape *size*, True inside the oval (foreground).
    """
    H, W = size
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    a = width_fraction  * (W - 1) / 2.0   # semi-axis x
    b = height_fraction * (H - 1) / 2.0   # semi-axis y

    y, x = np.ogrid[:H, :W]
    mask = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0
    return mask


def apply_black_mask(
    img: FloatImg,
    mask: BoolMask,
) -> FloatImg:
    """Set background pixels (mask == False) to 0 in *img*.

    Returns a copy; the input is not modified.
    """
    out = img.copy()
    out[~mask] = 0.0
    return out
