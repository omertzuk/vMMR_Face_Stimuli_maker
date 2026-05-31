"""Feature-based face morphing with boundary control points.

Algorithm
---------
1. Compute the intermediate landmark set:
       intermediate = (1 - alpha) * pts_self + alpha * pts_other
2. Append 8 boundary control points (4 corners + 4 edge midpoints) that are
   IDENTICAL in both images, ensuring the warp covers the full frame with no
   untriangulated boundary regions or seams.
3. Delaunay-triangulate the INTERMEDIATE point set.
4. For each triangle:
       a. warp self  -> intermediate triangle via affine (getAffineTransform)
       b. warp other -> intermediate triangle via affine
       c. blend: morph_tri = (1-alpha)*warp_self + alpha*warp_other
5. Composite all triangle patches into the output canvas.

IMPORTANT: cv2.getAffineTransform is used here intentionally — inside each
small triangle a full-6-DOF affine is the correct local warp (this is
different from the global similarity-only alignment step in align.py, which
must not introduce shear at the whole-image level).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import scipy.spatial
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatImg = NDArray[np.float64]


# ===========================================================================
# Public API
# ===========================================================================

def morph_pair(
    img_self: FloatImg,
    pts_self: NDArray[np.float32],
    img_other: FloatImg,
    pts_other: NDArray[np.float32],
    alpha: float = 0.5,
    out_size: tuple[int, int] = (425, 405),
) -> tuple[FloatImg, NDArray[np.float32]]:
    """Create a morphed image between *img_self* and *img_other*.

    Parameters
    ----------
    img_self, img_other:
        Float64 grayscale images of shape (H, W). Must have the same shape.
    pts_self, pts_other:
        (N, 2) float32 landmark arrays [x, y] in the aligned frame.
    alpha:
        Blending weight (0=self, 1=other; 0.5=50-50).
    out_size:
        (width, height) of the output canvas.

    Returns
    -------
    morph_img:   float64 (H, W) morphed image
    pts_morph:   (N+8, 2) intermediate landmark array (augmented with boundary)
    """
    if img_self.shape != img_other.shape:
        raise ValueError(
            f"Image shapes must match for morphing: {img_self.shape} vs {img_other.shape}"
        )

    W, H = out_size

    # Augment landmark sets with boundary control points
    boundary = _boundary_points(W, H)                      # (8, 2)
    aug_self = np.vstack([pts_self, boundary]).astype(np.float64)
    aug_other = np.vstack([pts_other, boundary]).astype(np.float64)

    # Intermediate landmark set
    intermediate = (1.0 - alpha) * aug_self + alpha * aug_other

    # Delaunay triangulation of intermediate points
    tri = scipy.spatial.Delaunay(intermediate)
    simplices = tri.simplices     # (T, 3) index array

    # Build output canvas
    morph_canvas = np.zeros((H, W), dtype=np.float64)

    for simplex in simplices:
        # Triangle vertices in each space
        tri_src_self  = aug_self[simplex].astype(np.float32)
        tri_src_other = aug_other[simplex].astype(np.float32)
        tri_dst       = intermediate[simplex].astype(np.float32)

        _warp_triangle(
            img_self, img_other, morph_canvas,
            tri_src_self, tri_src_other, tri_dst,
            alpha,
        )

    # pts_morph: the intermediate set (without boundary) for downstream use
    pts_morph = intermediate[: len(pts_self)].astype(np.float32)
    return morph_canvas, pts_morph


# ===========================================================================
# Core triangle warp
# ===========================================================================

def _warp_triangle(
    img_self: FloatImg,
    img_other: FloatImg,
    canvas: FloatImg,
    tri_self: NDArray[np.float32],
    tri_other: NDArray[np.float32],
    tri_dst: NDArray[np.float32],
    alpha: float,
) -> None:
    """Warp one triangle from self and other, blend, and paint onto canvas."""
    H, W = canvas.shape[:2]

    # Bounding rect of the destination triangle (clipped to canvas)
    rect = cv2.boundingRect(tri_dst.reshape(1, -1, 2))
    rx, ry, rw, rh = rect
    rx = max(rx, 0)
    ry = max(ry, 0)
    rw = min(rw, W - rx)
    rh = min(rh, H - ry)
    if rw <= 0 or rh <= 0:
        return

    # Localise vertices to the bounding rect
    tri_dst_local  = tri_dst  - np.array([rx, ry], dtype=np.float32)
    tri_self_local = tri_self
    tri_other_local = tri_other

    # Affine transforms: src -> dst-local  (6-DOF, correct for intra-triangle)
    M_self  = cv2.getAffineTransform(tri_self_local,  tri_dst_local)
    M_other = cv2.getAffineTransform(tri_other_local, tri_dst_local)

    # Warp source patches into the bounding rect
    patch_self  = cv2.warpAffine(
        img_self,  M_self,  (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    patch_other = cv2.warpAffine(
        img_other, M_other, (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Blend
    patch_morph = (1.0 - alpha) * patch_self + alpha * patch_other

    # Triangle mask within the bounding rect
    tri_mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, tri_dst_local.astype(np.int32), 255)
    tri_mask_bool = tri_mask.astype(bool)

    # Paint onto canvas
    roi = canvas[ry : ry + rh, rx : rx + rw]
    roi[tri_mask_bool] = patch_morph[tri_mask_bool]


# ===========================================================================
# Boundary points
# ===========================================================================

def _boundary_points(W: int, H: int) -> NDArray[np.float64]:
    """Return 8 boundary control points (corners + edge midpoints).

    These are IDENTICAL in both landmark sets, ensuring full-frame
    Delaunay triangulation with no untriangulated boundary regions.
    """
    return np.array([
        [0,       0],       # top-left
        [W - 1,   0],       # top-right
        [0,       H - 1],   # bottom-left
        [W - 1,   H - 1],   # bottom-right
        [W // 2,  0],       # top-edge midpoint
        [W // 2,  H - 1],   # bottom-edge midpoint
        [0,       H // 2],  # left-edge midpoint
        [W - 1,   H // 2],  # right-edge midpoint
    ], dtype=np.float64)
