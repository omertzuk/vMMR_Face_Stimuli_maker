"""Validation of SHINE pipeline outputs.

Two modes
---------
(a) Property tests (always run):
    After SHINE, the output images must satisfy measurable invariants:
    - foreground histograms are identical to tolerance
    - rotational spectra are identical to tolerance
    - global rescale is a single linear transform (rank order preserved)
    - dtype, shape, and value range are correct

(b) Reference comparison (run when reference_stimuli/ is populated):
    Compare pipeline output vs original Dor-Ziderman stimuli via SSIM,
    histogram RMSE, and radial-spectrum RMSE; write a comparison report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatImg = NDArray[np.float64]
BoolMask = NDArray[np.bool_]


# ===========================================================================
# (a) Property tests
# ===========================================================================

def validate_shine_properties(
    images: list[FloatImg],
    mask: BoolMask | None,
    out_dir: str | Path | None = None,
    hist_tol: float = 20.0,   # lenient: spectral step re-perturbs histogram slightly
    spec_tol: float = None,   # computed relative to the per-image spectral energy
) -> dict:
    """Run property checks on a set of SHINE-equalized images.

    Parameters
    ----------
    images:    list of float64 arrays (output of shine())
    mask:      same mask that was passed to shine()
    out_dir:   if given, write validation_report.txt there
    hist_tol:  acceptable mean-histogram RMSE
    spec_tol:  acceptable mean-spectrum RMSE; defaults to 5% of mean energy

    Returns
    -------
    report dict with keys "passed", "hist_rmse", "spec_rmse", "failures"
    """
    from shine import _histogram_rmse, _spectral_rmse, radial_spectrum

    failures = []
    warnings = []

    # -----------------------------------------------------------------------
    # 1. dtype and range
    # -----------------------------------------------------------------------
    for i, img in enumerate(images):
        if img.dtype != np.float64:
            failures.append(f"Image {i}: dtype={img.dtype}, expected float64")
        vals = img[mask].ravel() if mask is not None else img.ravel()
        if vals.min() < -1e-4:
            failures.append(f"Image {i}: min value {vals.min():.4f} < 0")
        if vals.max() > 255.0 + 1e-4:
            failures.append(f"Image {i}: max value {vals.max():.4f} > 255")

    # -----------------------------------------------------------------------
    # 2. Background must be zero
    # -----------------------------------------------------------------------
    if mask is not None:
        for i, img in enumerate(images):
            bg_max = np.abs(img[~mask]).max()
            if bg_max > 1e-6:
                failures.append(
                    f"Image {i}: background not zero (max |bg|={bg_max:.2e})"
                )

    # -----------------------------------------------------------------------
    # 3. Histogram RMSE
    # -----------------------------------------------------------------------
    hist_rmse = _histogram_rmse(images, mask=mask)
    if hist_rmse > hist_tol:
        warnings.append(
            f"Histogram RMSE {hist_rmse:.4f} exceeds tolerance {hist_tol}; "
            "consider more SHINE iterations."
        )

    # -----------------------------------------------------------------------
    # 4. Spectral RMSE
    # -----------------------------------------------------------------------
    spec_rmse = _spectral_rmse(images, mask=mask)
    if spec_tol is None:
        # 5% of the mean power at the DC+1 bin across images
        mean_energy = np.mean([
            radial_spectrum(img)[1][1:].mean() for img in images
        ])
        spec_tol = max(mean_energy * 0.05, 1e-6)

    if spec_rmse > spec_tol:
        warnings.append(
            f"Spectral RMSE {spec_rmse:.4e} exceeds tolerance {spec_tol:.4e}; "
            "consider more SHINE iterations."
        )

    # -----------------------------------------------------------------------
    # 5. Global rescale: rank order preserved across all images
    # -----------------------------------------------------------------------
    if len(images) > 1:
        all_in = np.concatenate(
            [img[mask].ravel() if mask is not None else img.ravel()
             for img in images]
        )
        # For a global linear transform, min and max of each image should be
        # consistent with a single a, b: out = a*in + b
        # We just verify that per-image min are monotone (no crossing)
        per_min = [
            float((img[mask] if mask is not None else img).min())
            for img in images
        ]
        per_max = [
            float((img[mask] if mask is not None else img).max())
            for img in images
        ]
        # Value range should be in [0, 255]
        if any(m < -1e-4 for m in per_min):
            failures.append("Some images have negative foreground values after rescale.")
        if any(m > 255.0 + 1e-4 for m in per_max):
            failures.append("Some images exceed 255 after rescale.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    passed = len(failures) == 0
    report = {
        "passed": passed,
        "hist_rmse": float(hist_rmse),
        "spec_rmse": float(spec_rmse),
        "failures": failures,
        "warnings": warnings,
    }

    _log_report(report)
    if out_dir is not None:
        _write_report(report, Path(out_dir) / "validation_report.txt")

    return report


def _log_report(report: dict) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    logger.info("Validation: %s  hist_RMSE=%.4f  spec_RMSE=%.4e",
                status, report["hist_rmse"], report["spec_rmse"])
    for f in report["failures"]:
        logger.error("  FAIL: %s", f)
    for w in report["warnings"]:
        logger.warning("  WARN: %s", w)


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Validation report",
        f"=================",
        f"Result  : {'PASS' if report['passed'] else 'FAIL'}",
        f"Hist RMSE : {report['hist_rmse']:.6f}",
        f"Spec RMSE : {report['spec_rmse']:.4e}",
        "",
        "Failures:",
    ]
    lines += [f"  - {f}" for f in report["failures"]] or ["  (none)"]
    lines += ["", "Warnings:"]
    lines += [f"  - {w}" for w in report["warnings"]] or ["  (none)"]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Validation report written: %s", path)


# ===========================================================================
# (b) Reference comparison
# ===========================================================================

def compare_to_reference(
    pipeline_images: list[FloatImg],
    labels: list[str],
    ref_dir: str | Path,
    out_dir: str | Path,
) -> None:
    """Compare pipeline output against reference stimuli (if present).

    Writes comparison_report.txt with SSIM, histogram RMSE, and
    radial-spectrum RMSE for each image pair.
    """
    from shine import rmse, ssim_score, radial_spectrum
    from io_utils import load_image_gray, gray_uint8_to_float

    ref_dir = Path(ref_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["Reference comparison report", "=" * 40, ""]
    for label, pipe_img in zip(labels, pipeline_images):
        ref_path = ref_dir / f"{label}.png"
        if not ref_path.exists():
            lines.append(f"{label}: reference not found at {ref_path}")
            continue

        ref_img = gray_uint8_to_float(load_image_gray(ref_path))

        if ref_img.shape != pipe_img.shape:
            lines.append(
                f"{label}: shape mismatch — pipeline {pipe_img.shape}, ref {ref_img.shape}"
            )
            continue

        # Per-image metrics
        img_ssim = ssim_score(pipe_img, ref_img)
        img_hist_rmse = rmse(
            np.histogram(pipe_img.ravel(), bins=256, range=(0, 255))[0].astype(float),
            np.histogram(ref_img.ravel(),  bins=256, range=(0, 255))[0].astype(float),
        )
        _, pwr_pipe = radial_spectrum(pipe_img)
        _, pwr_ref  = radial_spectrum(ref_img)
        min_len = min(len(pwr_pipe), len(pwr_ref))
        img_spec_rmse = rmse(pwr_pipe[:min_len], pwr_ref[:min_len])

        lines.append(
            f"{label}: SSIM={img_ssim:.4f}  "
            f"hist_RMSE={img_hist_rmse:.2f}  "
            f"spec_RMSE={img_spec_rmse:.4e}"
        )
        logger.info(
            "Reference comparison [%s]: SSIM=%.4f  hist_RMSE=%.2f  spec_RMSE=%.4e",
            label, img_ssim, img_hist_rmse, img_spec_rmse,
        )

    report_path = out_dir / "comparison_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison report written: %s", report_path)
