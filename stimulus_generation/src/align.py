"""Similarity alignment: rotate + uniform scale + translate only (no shear).

cv2.estimateAffinePartial2D gives a 4-DOF similarity transform.
cv2.getAffineTransform is deliberately NOT used here — it has 6 DOF and
introduces shear, which would distort face geometry.

The same transform is applied to both the image and the landmark array so
that morphing in morph.py can use the aligned landmarks directly.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ===========================================================================
# Template: fixed eye positions in the output frame
# ===========================================================================

def _compute_template_eyes(
    W: int,
    H: int,
    cfg: dict[str, Any],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return target left/right eye-centre positions in the (W, H) frame.

    'Left' and 'right' refer to the *person's* left/right (mirrored from
    the viewer). We place them symmetrically around the horizontal centre.
    """
    iod = cfg.get("interocular_distance", 140)   # pixels
    ey = cfg.get("eye_centre_y", 160)             # pixels from top

    cx = W / 2.0
    # person's left eye is to the viewer's right → higher x
    left_eye_x = cx + iod / 2.0
    right_eye_x = cx - iod / 2.0

    left = np.array([left_eye_x, ey], dtype=np.float32)
    right = np.array([right_eye_x, ey], dtype=np.float32)
    return left, right


# ===========================================================================
# Public API
# ===========================================================================

def align_to_template(
    img_bgr: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    W: int,
    H: int,
    cfg: dict[str, Any] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Similarity-align *img_bgr* so eyes land on the template positions.

    Parameters
    ----------
    img_bgr:    input colour image (BGR uint8)
    landmarks:  (N, 2) float32 landmark array [x, y]
    W, H:       output frame width and height in pixels
    cfg:        alignment config dict (interocular_distance, eye_centre_y)

    Returns
    -------
    aligned_img:  (H, W, 3) uint8 BGR image
    aligned_lms:  (N, 2) float32 landmarks in the aligned frame
    """
    if cfg is None:
        cfg = {}

    from landmarks import get_eye_centres

    # Detect landmark method from landmark count (heuristic)
    method = "mediapipe" if len(landmarks) > 68 else "fan"
    left_src, right_src = get_eye_centres(landmarks, method=method)

    left_dst, right_dst = _compute_template_eyes(W, H, cfg)

    # Build (2, N) arrays for estimateAffinePartial2D
    src_pts = np.stack([left_src, right_src], axis=0).astype(np.float32)
    dst_pts = np.stack([left_dst, right_dst], axis=0).astype(np.float32)

    # estimateAffinePartial2D = similarity: rotation + scale + translation
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if M is None:
        raise RuntimeError(
            "estimateAffinePartial2D failed to find a valid similarity transform. "
            "Check that the landmarks are correctly detected."
        )

    # Warp image
    aligned_img = cv2.warpAffine(
        img_bgr, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Apply the SAME transform to landmarks
    aligned_lms = _transform_landmarks(landmarks, M)

    _log_transform(M)
    return aligned_img, aligned_lms


# ===========================================================================
# Helpers
# ===========================================================================

def _transform_landmarks(
    landmarks: NDArray[np.float32],
    M: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Apply a 2×3 affine matrix to an (N, 2) landmark array."""
    # Homogeneous: (N, 3) × M.T → (N, 2)
    ones = np.ones((len(landmarks), 1), dtype=np.float32)
    pts_h = np.hstack([landmarks, ones])           # (N, 3)
    transformed = pts_h @ M.T                      # (N, 2)
    return transformed.astype(np.float32)


def _log_transform(M: NDArray[np.float64]) -> None:
    """Log the rotation angle and scale extracted from the similarity matrix."""
    a, b = M[0, 0], M[0, 1]
    scale = float(np.sqrt(a**2 + b**2))
    angle = float(np.degrees(np.arctan2(b, a)))
    tx, ty = float(M[0, 2]), float(M[1, 2])
    logger.debug(
        "Alignment: scale=%.3f  angle=%.2f°  tx=%.1f  ty=%.1f",
        scale, angle, tx, ty,
    )
