"""Tests for morph.py.

Run with:
    cd stimulus_generation
    python -m pytest tests/test_morph.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morph import morph_pair, _boundary_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H = 120, 100  # small for speed


def _flat_landmarks(n=20):
    """Distribute n points on a regular grid, well inside the frame."""
    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(10, W - 10, n),
        rng.uniform(10, H - 10, n),
    ]).astype(np.float32)
    return pts


def _constant_image(val: float) -> np.ndarray:
    return np.full((H, W), val, dtype=np.float64)


# ---------------------------------------------------------------------------
# _boundary_points
# ---------------------------------------------------------------------------

class TestBoundaryPoints:
    def test_count(self):
        bp = _boundary_points(W, H)
        assert len(bp) == 8

    def test_corners_included(self):
        bp = _boundary_points(W, H)
        corners = {(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)}
        bp_set = {(int(r[0]), int(r[1])) for r in bp}
        assert corners.issubset(bp_set)

    def test_within_frame(self):
        bp = _boundary_points(W, H)
        assert np.all(bp[:, 0] >= 0) and np.all(bp[:, 0] <= W - 1)
        assert np.all(bp[:, 1] >= 0) and np.all(bp[:, 1] <= H - 1)


# ---------------------------------------------------------------------------
# morph_pair
# ---------------------------------------------------------------------------

class TestMorphPair:
    def test_output_shape(self):
        pts = _flat_landmarks(30)
        img = _constant_image(128.0)
        morph, pts_m = morph_pair(img, pts, img, pts, alpha=0.5, out_size=(W, H))
        assert morph.shape == (H, W)
        assert pts_m.shape == (len(pts), 2)

    def test_alpha_zero_returns_self(self):
        """alpha=0 should return (approximately) the self image."""
        pts_self  = _flat_landmarks(20)
        pts_other = pts_self + 5.0   # slightly shifted
        img_self  = _constant_image(50.0)
        img_other = _constant_image(200.0)
        morph, _ = morph_pair(img_self, pts_self, img_other, pts_other,
                               alpha=0.0, out_size=(W, H))
        # Most interior pixels should equal img_self
        interior = morph[10:H-10, 10:W-10]
        assert np.abs(interior - 50.0).max() < 5.0

    def test_alpha_one_returns_other(self):
        pts_self  = _flat_landmarks(20)
        pts_other = pts_self + 5.0
        img_self  = _constant_image(50.0)
        img_other = _constant_image(200.0)
        morph, _ = morph_pair(img_self, pts_self, img_other, pts_other,
                               alpha=1.0, out_size=(W, H))
        interior = morph[10:H-10, 10:W-10]
        assert np.abs(interior - 200.0).max() < 5.0

    def test_alpha_half_midpoint_value(self):
        """For constant images, morph at alpha=0.5 must be the exact average."""
        pts = _flat_landmarks(30)
        img_self  = _constant_image(80.0)
        img_other = _constant_image(160.0)
        morph, _ = morph_pair(img_self, pts, img_other, pts,
                               alpha=0.5, out_size=(W, H))
        interior = morph[10:H-10, 10:W-10]
        assert np.abs(interior - 120.0).max() < 5.0

    def test_no_seams_uniform_images(self):
        """When both inputs are uniform and landmarks identical, output must be uniform."""
        pts = _flat_landmarks(30)
        val = 128.0
        img = _constant_image(val)
        morph, _ = morph_pair(img, pts, img, pts, alpha=0.5, out_size=(W, H))
        # Ignore 1-px border to allow for warpAffine boundary artefacts
        interior = morph[2:H-2, 2:W-2]
        assert np.abs(interior - val).max() < 2.0, "Seams detected in uniform-image morph"

    def test_dtype_float64(self):
        pts = _flat_landmarks(20)
        img = _constant_image(100.0)
        morph, _ = morph_pair(img, pts, img, pts)
        assert morph.dtype == np.float64

    def test_shape_mismatch_raises(self):
        pts = _flat_landmarks(20)
        img_a = np.zeros((H, W), dtype=np.float64)
        img_b = np.zeros((H + 10, W), dtype=np.float64)
        with pytest.raises(ValueError, match="shapes must match"):
            morph_pair(img_a, pts, img_b, pts)

    def test_full_frame_coverage(self):
        """The boundary control points must ensure all pixels are assigned."""
        pts_self  = _flat_landmarks(30)
        pts_other = pts_self + np.random.default_rng(5).uniform(-3, 3, pts_self.shape)
        img_self  = _constant_image(128.0)
        img_other = _constant_image(192.0)
        morph, _ = morph_pair(img_self, pts_self, img_other, pts_other,
                               alpha=0.5, out_size=(W, H))
        # Interior region should be entirely non-zero (no untriangulated voids)
        interior = morph[5:H-5, 5:W-5]
        assert np.all(interior > 0), "Untriangulated (zero) pixels found inside the frame"
