#!/usr/bin/env python
"""
Entry-point CLI for the face-stimuli generation pipeline.

Produces six 425×405 grayscale PNGs per participant:
    self.png  other.png  morph50.png
    self_target.png  other_target.png  morph50_target.png

Usage
-----
python stimulus_generation/create_face_stimuli.py \\
    --subject sub-001 \\
    --self  raw_faces/sub-001/self_raw.jpg \\
    --other raw_faces/sub-001/other_raw.jpg \\
    --out   stimuli/faces/sub-001 \\
    --landmark-method mediapipe \\
    --morph-alpha 0.35 \\
    --spectrum sf --iterations 10 \\
    --save-intermediates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of how the script is invoked
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from io_utils import (
    load_image_color,
    save_png,
    save_json,
    bgr_to_gray_float,
    float_to_uint8,
    ensure_output_dir,
)
from landmarks import detect_landmarks, load_landmarks_from_json, save_landmark_overlay
from align import align_to_template
from morph import morph_pair
from masks import create_face_mask, apply_black_mask
from shine import shine
from sunglasses import draw_sunglasses
from qc import (
    save_contact_sheet,
    save_histogram_plot,
    save_spectrum_plot,
    write_qc_csv,
)
from validate import validate_shine_properties, compare_to_reference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    if path.exists():
        with path.open() as fh:
            return yaml.safe_load(fh)
    return {}


def merge_config_args(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags override config.yaml values."""
    if args.spectrum is not None:
        cfg.setdefault("shine", {})["spectrum"] = args.spectrum
    if args.iterations is not None:
        cfg.setdefault("shine", {})["iterations"] = args.iterations
    if args.landmark_method is not None:
        cfg.setdefault("landmarks", {})["method"] = args.landmark_method
    if args.morph_alpha is not None:
        cfg.setdefault("morph", {})["alpha"] = args.morph_alpha
    if args.match_population is not None:
        cfg.setdefault("shine", {})["match_population"] = args.match_population
    cfg.setdefault("pipeline", {})["save_intermediates"] = args.save_intermediates
    cfg["pipeline"]["overwrite"] = args.overwrite
    cfg["pipeline"]["qc_only"] = args.qc_only
    return cfg


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate vMMR face stimuli for one participant."
    )
    p.add_argument("--subject", required=True, help="Participant ID (e.g. sub-001)")
    p.add_argument("--self", dest="self_img", required=True, help="Path to self face image")
    p.add_argument("--other", dest="other_img", required=True, help="Path to other face image")
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--landmark-method", choices=["mediapipe", "fan"], default=None)
    p.add_argument(
        "--morph-alpha",
        type=float,
        default=None,
        help="Morph blend weight: 0=self, 1=other (default: 0.5)",
    )
    p.add_argument("--spectrum", choices=["sf", "spec"], default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument(
        "--match-population",
        choices=["triplet", "set"],
        default=None,
        help="SHINE matching population: 'triplet' (default) or 'set'",
    )
    p.add_argument("--manual-landmarks", action="store_true",
                   help="Load landmarks from existing JSON instead of detecting")
    p.add_argument("--save-intermediates", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--qc-only", action="store_true",
                   help="Regenerate QC outputs only (stimuli must already exist)")
    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace, cfg: dict) -> None:
    out_dir = Path(args.out)
    ensure_output_dir(out_dir)

    overwrite = cfg["pipeline"]["overwrite"]
    save_intermediates = cfg["pipeline"]["save_intermediates"]
    qc_only = cfg["pipeline"]["qc_only"]

    lm_cfg = cfg.get("landmarks", {})
    morph_cfg = cfg.get("morph", {})
    shine_cfg = cfg.get("shine", {})
    mask_cfg = cfg.get("mask", {})
    output_cfg = cfg.get("output", {})

    W, H = output_cfg.get("size", [425, 405])
    landmark_method = lm_cfg.get("method", "mediapipe")

    # -----------------------------------------------------------------------
    # QC-only mode: reload stimuli and re-run QC
    # -----------------------------------------------------------------------
    if qc_only:
        _run_qc_only(out_dir, W, H)
        return

    # -----------------------------------------------------------------------
    # Step 1: Load raw colour images
    # -----------------------------------------------------------------------
    logger.info("[1] Loading raw images")
    img_self_bgr = load_image_color(args.self_img)
    img_other_bgr = load_image_color(args.other_img)

    # -----------------------------------------------------------------------
    # Step 2: Detect / load landmarks (colour images)
    # -----------------------------------------------------------------------
    logger.info("[2] Detecting landmarks")
    lm_dir = out_dir / "landmarks"
    lm_self_path = lm_dir / "landmarks_self.json"
    lm_other_path = lm_dir / "landmarks_other.json"

    if args.manual_landmarks:
        pts_self = np.array(load_landmarks_from_json(lm_self_path), dtype=np.float32)
        pts_other = np.array(load_landmarks_from_json(lm_other_path), dtype=np.float32)
        logger.info("  Loaded landmarks from JSON (manual mode)")
    else:
        pts_self = detect_landmarks(img_self_bgr, method=landmark_method)
        pts_other = detect_landmarks(img_other_bgr, method=landmark_method)
        ensure_output_dir(lm_dir)
        save_json(pts_self.tolist(), lm_self_path, overwrite=overwrite)
        save_json(pts_other.tolist(), lm_other_path, overwrite=overwrite)
        save_landmark_overlay(img_self_bgr, pts_self, lm_dir / "overlay_self.png", overwrite=overwrite)
        save_landmark_overlay(img_other_bgr, pts_other, lm_dir / "overlay_other.png", overwrite=overwrite)

    # -----------------------------------------------------------------------
    # Step 3: Similarity-align both images + transform landmarks
    # -----------------------------------------------------------------------
    logger.info("[3] Aligning to template (%dx%d)", W, H)
    align_cfg = cfg.get("alignment", {})
    img_self_aligned, pts_self_aligned = align_to_template(
        img_self_bgr, pts_self, W, H, align_cfg
    )
    img_other_aligned, pts_other_aligned = align_to_template(
        img_other_bgr, pts_other, W, H, align_cfg
    )
    if save_intermediates:
        inter = ensure_output_dir(out_dir / "intermediates")
        save_png(img_self_aligned, inter / "self_aligned.png", overwrite=overwrite)
        save_png(img_other_aligned, inter / "other_aligned.png", overwrite=overwrite)

    # -----------------------------------------------------------------------
    # Step 4: Convert to grayscale float64
    # -----------------------------------------------------------------------
    logger.info("[4] Converting to grayscale")
    gray_self = bgr_to_gray_float(img_self_aligned)
    gray_other = bgr_to_gray_float(img_other_aligned)

    # -----------------------------------------------------------------------
    # Step 5: Morph -> morph50 (pre-equalization)
    # -----------------------------------------------------------------------
    morph_alpha = float(morph_cfg.get("alpha", 0.5))
    logger.info("[5] Morphing (alpha=%.3f)", morph_alpha)
    gray_morph, pts_morph = morph_pair(
        gray_self, pts_self_aligned,
        gray_other, pts_other_aligned,
        alpha=morph_alpha, out_size=(W, H),
    )
    if save_intermediates:
        inter = ensure_output_dir(out_dir / "intermediates")
        save_png(float_to_uint8(gray_morph), inter / "morph50_pre_shine.png", overwrite=overwrite)

    # -----------------------------------------------------------------------
    # Step 6: Build common oval mask
    # -----------------------------------------------------------------------
    logger.info("[6] Building oval mask")
    mask = create_face_mask(
        size=(H, W),
        height_fraction=mask_cfg.get("height_fraction", 0.95),
        width_fraction=mask_cfg.get("width_fraction", 0.80),
    )

    # -----------------------------------------------------------------------
    # Step 7: SHINE-equalize {self, other, morph} together within mask
    # -----------------------------------------------------------------------
    logger.info("[7] SHINE equalization")
    images_in = [gray_self, gray_other, gray_morph]
    equalized = shine(
        images_in,
        mask=mask,
        do_hist=shine_cfg.get("do_hist", True),
        spectrum=shine_cfg.get("spectrum", "sf"),
        iterations=shine_cfg.get("iterations", 10),
        rescale_mode=shine_cfg.get("rescale_mode", "all_in_range"),
        rng=np.random.default_rng(42),
    )

    # -----------------------------------------------------------------------
    # Step 8: Apply mask as final clamp (bg=0)
    # -----------------------------------------------------------------------
    logger.info("[8] Applying mask (bg=0)")
    eq_self, eq_other, eq_morph = [apply_black_mask(im, mask) for im in equalized]

    # -----------------------------------------------------------------------
    # Step 9: Draw sunglasses on equalized images -> *_target.png
    # -----------------------------------------------------------------------
    logger.info("[9] Drawing sunglasses")
    sgl_cfg = cfg.get("sunglasses", {})
    target_self = draw_sunglasses(eq_self, pts_self_aligned, sgl_cfg)
    target_other = draw_sunglasses(eq_other, pts_other_aligned, sgl_cfg)
    target_morph = draw_sunglasses(eq_morph, pts_morph, sgl_cfg)

    # -----------------------------------------------------------------------
    # Step 10: Save six final PNGs + metadata
    # -----------------------------------------------------------------------
    logger.info("[10] Saving final stimuli")
    stim_dir = ensure_output_dir(out_dir / "stimuli")

    outputs = {
        "self.png":          eq_self,
        "other.png":         eq_other,
        "morph50.png":       eq_morph,
        "self_target.png":   target_self,
        "other_target.png":  target_other,
        "morph50_target.png": target_morph,
    }
    for fname, arr in outputs.items():
        save_png(float_to_uint8(arr), stim_dir / fname, overwrite=overwrite)

    metadata = {
        "subject": args.subject,
        "self_image": str(args.self_img),
        "other_image": str(args.other_img),
        "output_size": [W, H],
        "landmark_method": landmark_method,
        "morph": morph_cfg,
        "shine": shine_cfg,
        "mask": mask_cfg,
        "alignment": align_cfg,
        "sunglasses": sgl_cfg,
    }
    save_json(metadata, out_dir / "processing_metadata.json", overwrite=overwrite)

    # -----------------------------------------------------------------------
    # Step 11: QC
    # -----------------------------------------------------------------------
    logger.info("[11] Generating QC outputs")
    eq_images = [eq_self, eq_other, eq_morph]
    labels = ["self", "other", "morph50"]
    qc_dir = ensure_output_dir(out_dir / "qc")

    save_contact_sheet(
        list(outputs.values()),
        list(outputs.keys()),
        qc_dir / "contact_sheet.png",
        overwrite=overwrite,
    )
    save_histogram_plot(eq_images, labels, mask, qc_dir / "histograms.png", overwrite=overwrite)
    save_spectrum_plot(eq_images, labels, qc_dir / "spectra.png", overwrite=overwrite)
    write_qc_csv(eq_images, labels, mask, qc_dir / "qc_report.csv", overwrite=overwrite)

    validate_shine_properties(eq_images, mask, out_dir=qc_dir)

    ref_dir = Path(__file__).parent / "reference_stimuli"
    if any(ref_dir.iterdir()) if ref_dir.exists() else False:
        compare_to_reference(eq_images, labels, ref_dir, qc_dir)

    logger.info("Done. Stimuli saved to %s", stim_dir)


def _run_qc_only(out_dir: Path, W: int, H: int) -> None:
    from io_utils import load_image_gray, gray_uint8_to_float
    stim_dir = out_dir / "stimuli"
    names = ["self", "other", "morph50"]
    images = [gray_uint8_to_float(load_image_gray(stim_dir / f"{n}.png")) for n in names]
    mask = create_face_mask(size=(H, W))
    qc_dir = ensure_output_dir(out_dir / "qc")
    save_histogram_plot(images, names, mask, qc_dir / "histograms.png", overwrite=True)
    save_spectrum_plot(images, names, qc_dir / "spectra.png", overwrite=True)
    write_qc_csv(images, names, mask, qc_dir / "qc_report.csv", overwrite=True)
    validate_shine_properties(images, mask, out_dir=qc_dir)
    logger.info("QC-only done.")


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
    cfg = merge_config_args(cfg, args)
    run_pipeline(args, cfg)


if __name__ == "__main__":
    main()
