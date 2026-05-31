"""Draw opaque black sunglasses on an already-equalized face image.

Target images are produced by overlaying sunglasses on the equalized
self / other / morph50 images. SHINE is NOT run on targets.

Geometry is derived from eye landmarks detected during the alignment step.
The same geometry routine is applied to all three images to guarantee
consistent-looking sunglasses across the set.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from landmarks import _MP_LEFT_EYE_IDXS, _MP_RIGHT_EYE_IDXS

logger = logging.getLogger(__name__)

FloatImg = NDArray[np.float64]


def draw_sunglasses(
    img: FloatImg,
    landmarks: NDArray[np.float32],
    cfg: dict[str, Any] | None = None,
) -> FloatImg:
    """Draw opaque black sunglasses on *img*.

    Sunglasses are drawn ONLY on the equalized stimulus image. Landmarks must
    be in the aligned coordinate frame (i.e., post align_to_template).

    Parameters
    ----------
    img:
        Float64 grayscale image in [0, 255].
    landmarks:
        (N, 2) float32 aligned landmark array.  Supports both 478-pt
        (MediaPipe) and 68-pt (FAN) sets.
    cfg:
        Optional config dict with keys:
            lens_scale        (default 1.6) — width multiplier relative to eye width
            lens_height_frac  (default 0.55) — lens height / lens width
            bridge_height     (default 6) — bridge thickness in pixels

    Returns
    -------
    result : float64 copy of *img* with sunglasses painted in black (0).
    """
    if cfg is None:
        cfg = {}

    lens_scale       = float(cfg.get("lens_scale",       1.6))
    lens_height_frac = float(cfg.get("lens_height_frac", 0.55))
    bridge_height    = int(cfg.get("bridge_height",       6))

    result = img.copy()
    canvas = result  # work in-place on the copy

    # Detect landmark set size
    n_pts = len(landmarks)
    if n_pts >= 400:
        left_eye_pts  = landmarks[_MP_LEFT_EYE_IDXS]
        right_eye_pts = landmarks[_MP_RIGHT_EYE_IDXS]
    else:
        # FAN 68-pt
        left_eye_pts  = landmarks[36:42]
        right_eye_pts = landmarks[42:48]

    left_centre  = left_eye_pts.mean(axis=0)
    right_centre = right_eye_pts.mean(axis=0)

    # Half-width of each eye region (span of eye points)
    left_hw  = _eye_half_width(left_eye_pts)
    right_hw = _eye_half_width(right_eye_pts)

    # Draw left lens, right lens, then bridge
    _draw_lens(canvas, left_centre,  left_hw,  lens_scale, lens_height_frac)
    _draw_lens(canvas, right_centre, right_hw, lens_scale, lens_height_frac)
    _draw_bridge(canvas, left_centre, right_centre, bridge_height)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eye_half_width(eye_pts: NDArray[np.float32]) -> float:
    """Horizontal half-span of an eye landmark cluster."""
    xs = eye_pts[:, 0]
    return float((xs.max() - xs.min()) / 2.0)


def _draw_lens(
    canvas: FloatImg,
    centre: NDArray[np.float32],
    half_width: float,
    scale: float,
    height_frac: float,
) -> None:
    """Paint an opaque black ellipse for one lens."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    ax = int(round(half_width * scale))
    ay = int(round(ax * height_frac))
    cv2.ellipse(
        canvas,
        (cx, cy),
        (max(ax, 1), max(ay, 1)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=0.0,
        thickness=-1,   # filled
    )


def _draw_bridge(
    canvas: FloatImg,
    left_centre: NDArray[np.float32],
    right_centre: NDArray[np.float32],
    height: int,
) -> None:
    """Paint an opaque black rectangle bridging the two lenses."""
    lx = int(round(left_centre[0]))
    rx = int(round(right_centre[0]))
    y  = int(round((left_centre[1] + right_centre[1]) / 2.0))
    hh = max(height // 2, 1)

    # The person's right eye centre is at higher x; left is at lower x.
    x_min = min(lx, rx)
    x_max = max(lx, rx)

    cv2.rectangle(
        canvas,
        (x_min, y - hh),
        (x_max, y + hh),
        color=0.0,
        thickness=-1,
    )
