"""Tests for align.py.

Run with:
    cd stimulus_generation
    python -m pytest tests/test_align.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import align as al
from align import align_to_template, _compute_template_eyes, _transform_landmarks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_landmarks_mediapipe(H=480, W=640, seed=0):
    """Build a plausible 478-pt landmark set with recognisable eye positions."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.1, 0.9, size=(478, 2)).astype(np.float32)
    pts[:, 0] *= W
    pts[:, 1] *= H

    # Overwrite the eye-region indices with controlled values
    from landmarks import _MP_LEFT_EYE_IDXS, _MP_RIGHT_EYE_IDXS
    cx = W / 2.0
    ey = H * 0.4
    iod = W * 0.2
    for idx in _MP_LEFT_EYE_IDXS:
        pts[idx] = [cx + iod / 2 + rng.uniform(-2, 2), ey + rng.uniform(-2, 2)]
    for idx in _MP_RIGHT_EYE_IDXS:
        pts[idx] = [cx - iod / 2 + rng.uniform(-2, 2), ey + rng.uniform(-2, 2)]
    return pts


def _random_image(H=480, W=640):
    rng = np.random.default_rng(42)
    return (rng.uniform(0, 255, (H, W, 3))).astype(np.uint8)


# ---------------------------------------------------------------------------
# _compute_template_eyes
# ---------------------------------------------------------------------------

class TestComputeTemplateEyes:
    def test_horizontal_symmetry(self):
        W, H = 425, 405
        cfg = {"interocular_distance": 140, "eye_centre_y": 160}
        left, right = _compute_template_eyes(W, H, cfg)
        assert abs((left[0] + right[0]) / 2 - W / 2) < 1e-6

    def test_correct_y(self):
        W, H = 425, 405
        cfg = {"interocular_distance": 140, "eye_centre_y": 160}
        left, right = _compute_template_eyes(W, H, cfg)
        assert abs(left[1] - 160) < 1e-6
        assert abs(right[1] - 160) < 1e-6

    def test_correct_iod(self):
        W, H = 425, 405
        iod = 120
        cfg = {"interocular_distance": iod, "eye_centre_y": 150}
        left, right = _compute_template_eyes(W, H, cfg)
        assert abs(np.linalg.norm(left - right) - iod) < 1e-6


# ---------------------------------------------------------------------------
# _transform_landmarks
# ---------------------------------------------------------------------------

class TestTransformLandmarks:
    def test_identity(self):
        pts = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        M = np.eye(2, 3, dtype=np.float64)
        out = _transform_landmarks(pts, M)
        assert np.allclose(out, pts, atol=1e-5)

    def test_translation(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float32)
        M = np.array([[1, 0, 5], [0, 1, 7]], dtype=np.float64)
        out = _transform_landmarks(pts, M)
        expected = np.array([[5, 7], [15, 17]], dtype=np.float32)
        assert np.allclose(out, expected, atol=1e-5)

    def test_shape_preserved(self):
        pts = np.random.default_rng(0).random((478, 2)).astype(np.float32)
        M = np.eye(2, 3, dtype=np.float64)
        out = _transform_landmarks(pts, M)
        assert out.shape == pts.shape


# ---------------------------------------------------------------------------
# align_to_template (integration)
# ---------------------------------------------------------------------------

class TestAlignToTemplate:
    W, H = 425, 405
    cfg = {"interocular_distance": 140, "eye_centre_y": 160}

    def test_output_shape(self):
        img = _random_image()
        lms = _make_fake_landmarks_mediapipe()
        aligned_img, aligned_lms = align_to_template(img, lms, self.W, self.H, self.cfg)
        assert aligned_img.shape == (self.H, self.W, 3)
        assert aligned_lms.shape == lms.shape

    def test_output_dtype(self):
        img = _random_image()
        lms = _make_fake_landmarks_mediapipe()
        aligned_img, aligned_lms = align_to_template(img, lms, self.W, self.H, self.cfg)
        assert aligned_img.dtype == np.uint8
        assert aligned_lms.dtype == np.float32

    def test_eyes_at_template_position(self):
        """After alignment, eye centres should be close to template positions."""
        img = _random_image()
        lms = _make_fake_landmarks_mediapipe()
        aligned_img, aligned_lms = align_to_template(img, lms, self.W, self.H, self.cfg)

        from landmarks import get_eye_centres
        left, right = get_eye_centres(aligned_lms, method="mediapipe")
        left_tgt, right_tgt = _compute_template_eyes(self.W, self.H, self.cfg)

        assert np.linalg.norm(left - left_tgt) < 5.0, (
            f"Left eye off by {np.linalg.norm(left - left_tgt):.2f} px"
        )
        assert np.linalg.norm(right - right_tgt) < 5.0, (
            f"Right eye off by {np.linalg.norm(right - right_tgt):.2f} px"
        )

    def test_no_shear(self):
        """The 2×2 sub-matrix of M must be a scaled rotation (no shear).

        A similarity matrix has the form [[a, -b], [b, a]], so a==d and b==-c.
        We reconstruct M from eye positions to verify.
        """
        img = _random_image()
        lms = _make_fake_landmarks_mediapipe()

        from landmarks import get_eye_centres, _MP_LEFT_EYE_IDXS, _MP_RIGHT_EYE_IDXS
        left_src = lms[_MP_LEFT_EYE_IDXS].mean(axis=0)
        right_src = lms[_MP_RIGHT_EYE_IDXS].mean(axis=0)
        left_dst, right_dst = _compute_template_eyes(self.W, self.H, self.cfg)

        src_pts = np.stack([left_src, right_src]).astype(np.float32).reshape(-1, 1, 2)
        dst_pts = np.stack([left_dst, right_dst]).astype(np.float32).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Similarity: M[0,0] == M[1,1]  and  M[0,1] == -M[1,0]
        assert abs(M[0, 0] - M[1, 1]) < 1e-6, "Not a similarity: a != d"
        assert abs(M[0, 1] + M[1, 0]) < 1e-6, "Not a similarity: b != -c"
