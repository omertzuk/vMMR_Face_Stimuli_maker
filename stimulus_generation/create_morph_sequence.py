#!/usr/bin/env python
"""Generate an intermediate morph sequence between two face images.

Source of truth
---------------
The actual morphing implementation lives in ``src/morph.py:morph_pair``.  This
file is only orchestration: it follows the same loading, landmark detection,
similarity alignment, grayscale conversion, oval masking, and PNG output
helpers used by ``create_face_stimuli.py``.

Sequence convention
-------------------
``--num-intermediates N`` creates exactly N morph PNGs.  These morph PNGs are
intermediate-only and exclude the endpoints.  The alpha step is
``1 / (N + 1)``, so the generated morph alphas are:

    step, 2 * step, ..., N * step

For example, ``--num-intermediates 3`` produces alphas 0.25, 0.50, and 0.75.
The aligned/masked endpoints are saved separately as ``self.png`` and
``other.png``.  If video output is enabled, the video frame order is
``self.png``, then the intermediate morph PNGs, then ``other.png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of how the script is invoked
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from align import align_to_template
from io_utils import (
    bgr_to_gray_float,
    ensure_output_dir,
    float_to_uint8,
    load_image_color,
    save_json,
    save_png,
)
from landmarks import detect_landmarks, load_landmarks_from_json, save_landmark_overlay
from masks import apply_black_mask, create_face_mask
from morph import morph_pair

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class SequenceFrame:
    """One saved frame in the full endpoint-inclusive sequence."""

    label: str
    alpha: float
    path: Path
    image: np.ndarray


# ---------------------------------------------------------------------------
# Config and deterministic sequence helpers
# ---------------------------------------------------------------------------

def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    if path.exists():
        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


def compute_intermediate_alphas(num_intermediates: int) -> list[float]:
    """Return deterministic intermediate-only alpha values.

    The endpoints alpha=0 and alpha=1 are deliberately excluded.  For N
    intermediates the step is 1 / (N + 1), yielding N alphas inside (0, 1).
    """
    if num_intermediates < 0:
        raise ValueError("num_intermediates must be >= 0")

    denominator = num_intermediates + 1
    return [idx / denominator for idx in range(1, num_intermediates + 1)]


def video_timing(num_frames: int, total_duration: float) -> tuple[float, float]:
    """Return ``(seconds_per_frame, fps)`` for one-write-per-image video."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if total_duration <= 0:
        raise ValueError("total_duration must be > 0")

    seconds_per_frame = total_duration / num_frames
    fps = num_frames / total_duration
    return seconds_per_frame, fps


def build_full_video_sequence(
    self_frame: SequenceFrame,
    morph_frames: Sequence[SequenceFrame],
    other_frame: SequenceFrame,
) -> list[SequenceFrame]:
    """Return the endpoint-inclusive frame order used for video output."""
    return [self_frame, *morph_frames, other_frame]


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _morph_filename(index: int, alpha: float, total: int) -> str:
    width = max(3, len(str(total)))
    return f"morph_{index:0{width}d}_alpha_{alpha:.3f}.png"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate an intermediate-only face morph sequence. "
            "The N morph PNGs exclude endpoints; alphas are i/(N+1)."
        )
    )
    p.add_argument("--self", dest="self_img", required=True, help="Path to self face image")
    p.add_argument("--other", dest="other_img", required=True, help="Path to other face image")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--num-intermediates",
        type=_nonnegative_int,
        required=True,
        help=(
            "Number of intermediate morph PNGs to create. Endpoints are not "
            "counted; alpha step is 1/(N+1)."
        ),
    )
    p.add_argument("--landmark-method", choices=["mediapipe", "fan"], default=None)
    p.add_argument(
        "--manual-landmarks",
        action="store_true",
        help=(
            "Load landmarks from OUT/landmarks/landmarks_self.json and "
            "landmarks_other.json instead of detecting."
        ),
    )
    p.add_argument(
        "--video",
        action="store_true",
        help=(
            "Save morph_sequence.mp4 using the full sequence: self, "
            "intermediate morphs, other."
        ),
    )
    p.add_argument(
        "--video-duration",
        type=_positive_float,
        default=2.0,
        help=(
            "Total video duration in seconds for self + morphs + other. "
            "The writer FPS is frame_count / duration. Default: 2.0"
        ),
    )
    p.add_argument(
        "--video-name",
        default="morph_sequence.mp4",
        help="Video filename saved in the output directory when --video is set.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_sequence(args: argparse.Namespace, cfg: dict[str, Any]) -> list[SequenceFrame]:
    out_dir = ensure_output_dir(args.out)

    lm_cfg = cfg.get("landmarks", {})
    align_cfg = cfg.get("alignment", {})
    mask_cfg = cfg.get("mask", {})
    output_cfg = cfg.get("output", {})

    W, H = output_cfg.get("size", [425, 405])
    landmark_method = args.landmark_method or lm_cfg.get("method", "mediapipe")
    alphas = compute_intermediate_alphas(args.num_intermediates)
    alpha_step = 1.0 / (args.num_intermediates + 1)

    logger.info("Output size: %dx%d", W, H)
    logger.info(
        "Intermediate-only alpha sequence: %s",
        ", ".join(f"{alpha:.3f}" for alpha in alphas) if alphas else "(none)",
    )

    # -----------------------------------------------------------------------
    # Step 1: Load raw colour images
    # -----------------------------------------------------------------------
    logger.info("[1] Loading raw images")
    img_self_bgr = load_image_color(args.self_img)
    img_other_bgr = load_image_color(args.other_img)

    # -----------------------------------------------------------------------
    # Step 2: Detect / load landmarks on raw colour images
    # -----------------------------------------------------------------------
    logger.info("[2] Loading landmarks" if args.manual_landmarks else "[2] Detecting landmarks")
    lm_dir = out_dir / "landmarks"
    lm_self_path = lm_dir / "landmarks_self.json"
    lm_other_path = lm_dir / "landmarks_other.json"

    if args.manual_landmarks:
        pts_self = np.array(load_landmarks_from_json(lm_self_path), dtype=np.float32)
        pts_other = np.array(load_landmarks_from_json(lm_other_path), dtype=np.float32)
    else:
        pts_self = detect_landmarks(img_self_bgr, method=landmark_method)
        pts_other = detect_landmarks(img_other_bgr, method=landmark_method)
        ensure_output_dir(lm_dir)
        save_json(pts_self.tolist(), lm_self_path, overwrite=args.overwrite)
        save_json(pts_other.tolist(), lm_other_path, overwrite=args.overwrite)
        save_landmark_overlay(
            img_self_bgr, pts_self, lm_dir / "overlay_self.png", overwrite=args.overwrite
        )
        save_landmark_overlay(
            img_other_bgr, pts_other, lm_dir / "overlay_other.png", overwrite=args.overwrite
        )

    # -----------------------------------------------------------------------
    # Step 3: Similarity-align both images + transform landmarks
    # -----------------------------------------------------------------------
    logger.info("[3] Aligning to template")
    img_self_aligned, pts_self_aligned = align_to_template(
        img_self_bgr, pts_self, W, H, align_cfg
    )
    img_other_aligned, pts_other_aligned = align_to_template(
        img_other_bgr, pts_other, W, H, align_cfg
    )

    # -----------------------------------------------------------------------
    # Step 4: Convert to grayscale float64
    # -----------------------------------------------------------------------
    logger.info("[4] Converting to grayscale")
    gray_self = bgr_to_gray_float(img_self_aligned)
    gray_other = bgr_to_gray_float(img_other_aligned)

    # -----------------------------------------------------------------------
    # Step 5: Build the same oval mask convention used by the main pipeline
    # -----------------------------------------------------------------------
    logger.info("[5] Building common oval mask")
    mask = create_face_mask(
        size=(H, W),
        height_fraction=mask_cfg.get("height_fraction", 0.95),
        width_fraction=mask_cfg.get("width_fraction", 0.70),
    )

    # -----------------------------------------------------------------------
    # Step 6: Save masked endpoints and intermediate morphs
    # -----------------------------------------------------------------------
    logger.info("[6] Morphing and saving sequence")
    self_uint8 = float_to_uint8(apply_black_mask(gray_self, mask))
    other_uint8 = float_to_uint8(apply_black_mask(gray_other, mask))

    self_path = save_png(self_uint8, out_dir / "self.png", overwrite=args.overwrite)
    self_frame = SequenceFrame("self", 0.0, self_path, self_uint8)

    morph_frames: list[SequenceFrame] = []
    for idx, alpha in enumerate(alphas, start=1):
        logger.info("  Morph %d/%d (alpha=%.6f)", idx, len(alphas), alpha)
        gray_morph, _ = morph_pair(
            gray_self,
            pts_self_aligned,
            gray_other,
            pts_other_aligned,
            alpha=alpha,
            out_size=(W, H),
        )
        morph_uint8 = float_to_uint8(apply_black_mask(gray_morph, mask))
        morph_path = save_png(
            morph_uint8,
            out_dir / _morph_filename(idx, alpha, args.num_intermediates),
            overwrite=args.overwrite,
        )
        morph_frames.append(SequenceFrame("morph", float(alpha), morph_path, morph_uint8))

    other_path = save_png(other_uint8, out_dir / "other.png", overwrite=args.overwrite)
    other_frame = SequenceFrame("other", 1.0, other_path, other_uint8)

    full_sequence = build_full_video_sequence(self_frame, morph_frames, other_frame)

    video_info: dict[str, Any] | None = None
    if args.video:
        video_path = out_dir / Path(args.video_name).name
        video_info = write_sequence_video(
            [frame.image for frame in full_sequence],
            video_path,
            total_duration=args.video_duration,
            overwrite=args.overwrite,
        )
        logger.info("Video saved to %s", video_path)

    metadata = {
        "self_image": str(args.self_img),
        "other_image": str(args.other_img),
        "output_size": [W, H],
        "landmark_method": landmark_method,
        "sequence_convention": (
            "num_intermediates creates intermediate morph PNGs only; endpoints "
            "are saved as self.png and other.png and are included in video."
        ),
        "num_intermediates": args.num_intermediates,
        "alpha_step": alpha_step,
        "intermediate_alphas": [float(alpha) for alpha in alphas],
        "frames": [
            {
                "label": frame.label,
                "alpha": frame.alpha,
                "path": str(frame.path.name),
            }
            for frame in full_sequence
        ],
        "mask": mask_cfg,
        "alignment": align_cfg,
        "video": video_info,
    }
    save_json(metadata, out_dir / "morph_sequence_metadata.json", overwrite=args.overwrite)

    logger.info("Done. Sequence saved to %s", out_dir)
    return full_sequence


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def write_sequence_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    total_duration: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write frames to a video, one image per encoded frame.

    FPS is derived from the requested duration as ``len(frames) / duration``.
    Therefore each saved PNG in the full sequence has the same display time:
    ``duration / len(frames)`` seconds.
    """
    if not frames:
        raise ValueError("At least one frame is required to write a video")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = _as_bgr_uint8(frames[0])
    H, W = first.shape[:2]
    seconds_per_frame, fps = video_timing(len(frames), total_duration)

    logger.info(
        "Writing %d video frames at %.6f fps (%.6f s/frame)",
        len(frames),
        fps,
        seconds_per_frame,
    )
    writer = _open_video_writer(output_path, fps, frame_size=(W, H))
    try:
        writer.write(first)
        for frame in frames[1:]:
            frame_bgr = _as_bgr_uint8(frame)
            if frame_bgr.shape[:2] != (H, W):
                raise ValueError(
                    "All video frames must have the same shape: "
                    f"{frame_bgr.shape[:2]} vs {(H, W)}"
                )
            writer.write(frame_bgr)
    finally:
        writer.release()

    return {
        "path": str(output_path.name),
        "frame_count": len(frames),
        "total_duration": float(total_duration),
        "seconds_per_frame": float(seconds_per_frame),
        "fps": float(fps),
    }


def _as_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = float_to_uint8(frame)

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Expected grayscale or BGR frame, got shape {frame.shape}")


def _open_video_writer(
    output_path: Path,
    fps: float,
    *,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    suffix = output_path.suffix.lower()
    codecs = ["mp4v", "avc1"] if suffix == ".mp4" else ["MJPG", "XVID", "mp4v"]

    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size, True)
        if writer.isOpened():
            logger.debug("Using video codec %s for %s", codec, output_path)
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not open video writer for {output_path}. "
        "Try a different --video-name extension such as .avi."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()
    run_sequence(args, cfg)


if __name__ == "__main__":
    main()
