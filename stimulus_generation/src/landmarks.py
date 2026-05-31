"""Landmark detection for face images.

Primary detector : MediaPipe Face Landmarker (478-pt mesh, no torch dep).
Fallback detector: face-alignment (68-pt FAN, requires torch).

Contract
--------
- Exactly ONE face must be present; functions raise ValueError otherwise.
- Landmarks are returned as float32 (N, 2) arrays in pixel coordinates
  [x, y] where x is the column and y is the row.
- JSON format: list of [x, y] pairs.
- QC overlay PNGs are saved alongside the JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from io_utils import save_png

logger = logging.getLogger(__name__)

LandmarkMethod = Literal["mediapipe", "fan"]


# ===========================================================================
# Public API
# ===========================================================================

def detect_landmarks(
    img_bgr: NDArray[np.uint8],
    method: LandmarkMethod = "mediapipe",
) -> NDArray[np.float32]:
    """Detect face landmarks in *img_bgr* (BGR uint8).

    Returns (N, 2) float32 array of [x, y] pixel coordinates.
    Raises ValueError if the number of detected faces != 1.
    """
    if method == "mediapipe":
        return _detect_mediapipe(img_bgr)
    elif method == "fan":
        return _detect_fan(img_bgr)
    else:
        raise ValueError(f"Unknown landmark method: {method!r}")


def load_landmarks_from_json(path: str | Path) -> NDArray[np.float32]:
    """Load landmarks from a JSON file (list of [x, y] pairs).

    This is the --manual-landmarks path: corrected JSON is the canonical
    reproducible artifact. A drag-and-drop editor can write the same format.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Landmark JSON not found: {path}\n"
            "Run without --manual-landmarks first to generate it."
        )
    with path.open() as fh:
        data = json.load(fh)
    return np.array(data, dtype=np.float32)


def save_landmark_overlay(
    img_bgr: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    path: str | Path,
    *,
    overwrite: bool = False,
    dot_radius: int = 2,
) -> None:
    """Draw landmarks on a copy of *img_bgr* and save as PNG."""
    overlay = img_bgr.copy()
    for x, y in landmarks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), dot_radius, (0, 255, 0), -1)
    save_png(overlay, path, overwrite=overwrite)


# ===========================================================================
# Eye-centre helpers  (used by align.py)
# ===========================================================================

def get_eye_centres(
    landmarks: NDArray[np.float32],
    method: LandmarkMethod = "mediapipe",
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return (left_eye_centre, right_eye_centre) as (x, y) float32 arrays.

    From the viewer's perspective:
        left_eye  = person's right eye (higher x in standard layout)
        right_eye = person's left eye  (lower x)
    We follow the MediaPipe / FAN convention where left/right refer to the
    *person's* left/right.
    """
    if method == "mediapipe":
        return _eye_centres_mediapipe(landmarks)
    else:
        return _eye_centres_fan(landmarks)


# ===========================================================================
# MediaPipe detector
# ===========================================================================

# MediaPipe 478-landmark face mesh eye indices
# Left eye (person's left) outer/inner corners + lid points
_MP_LEFT_EYE_IDXS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
_MP_RIGHT_EYE_IDXS = [
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398,
]


def _detect_mediapipe(img_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for landmark detection. "
            "Install it with: pip install mediapipe"
        ) from exc

    mp_face_mesh = mp.solutions.face_mesh
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,       # detect up to 2 so we can error on >1
        min_detection_confidence=0.5,
        refine_landmarks=True,
    ) as face_mesh:
        result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No face detected in the image.")
    if len(result.multi_face_landmarks) > 1:
        raise ValueError(
            f"Expected exactly 1 face, found {len(result.multi_face_landmarks)}. "
            "Crop the image so only one face is visible."
        )

    face = result.multi_face_landmarks[0]
    pts = np.array(
        [[lm.x * W, lm.y * H] for lm in face.landmark],
        dtype=np.float32,
    )
    logger.debug("MediaPipe: detected %d landmarks", len(pts))
    return pts


def _eye_centres_mediapipe(
    landmarks: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    left = landmarks[_MP_LEFT_EYE_IDXS].mean(axis=0)
    right = landmarks[_MP_RIGHT_EYE_IDXS].mean(axis=0)
    return left, right


# ===========================================================================
# FAN (face-alignment) fallback
# ===========================================================================

# 68-point FAN indices for eye regions
_FAN_LEFT_EYE_IDXS = list(range(36, 42))   # left eye corners + lids
_FAN_RIGHT_EYE_IDXS = list(range(42, 48))  # right eye corners + lids


def _detect_fan(img_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    try:
        import face_alignment
    except ImportError as exc:
        raise ImportError(
            "face_alignment is required for FAN detection. "
            "Install with: pip install face-alignment torch torchvision"
        ) from exc

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cpu",
    )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(img_rgb)

    if preds is None or len(preds) == 0:
        raise ValueError("No face detected (FAN).")
    if len(preds) > 1:
        raise ValueError(
            f"Expected exactly 1 face (FAN), found {len(preds)}. "
            "Crop the image so only one face is visible."
        )

    pts = preds[0].astype(np.float32)   # shape (68, 2)
    logger.debug("FAN: detected %d landmarks", len(pts))
    return pts


def _eye_centres_fan(
    landmarks: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    left = landmarks[_FAN_LEFT_EYE_IDXS].mean(axis=0)
    right = landmarks[_FAN_RIGHT_EYE_IDXS].mean(axis=0)
    return left, right
