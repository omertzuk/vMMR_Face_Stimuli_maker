"""Quality-control outputs: contact sheet, histogram plots, spectrum plots, CSV."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatImg = NDArray[np.float64]
BoolMask = NDArray[np.bool_]


# ===========================================================================
# Contact sheet
# ===========================================================================

def save_contact_sheet(
    images: list[FloatImg | NDArray[np.uint8]],
    labels: list[str],
    path: str | Path,
    *,
    ncols: int = 3,
    overwrite: bool = False,
    border: int = 4,
) -> None:
    """Tile *images* into a contact-sheet PNG with labels."""
    from io_utils import save_png, float_to_uint8

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    uint8_imgs = []
    for img in images:
        if img.dtype != np.uint8:
            u = np.clip(img, 0, 255).astype(np.uint8)
        else:
            u = img
        # Convert to 3-ch for labelling
        if u.ndim == 2:
            u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
        uint8_imgs.append(u)

    H, W = uint8_imgs[0].shape[:2]
    nrows = (len(uint8_imgs) + ncols - 1) // ncols

    sheet_h = nrows * (H + border) + border
    sheet_w = ncols * (W + border) + border
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(uint8_imgs, labels)):
        row = idx // ncols
        col = idx % ncols
        y0 = row * (H + border) + border
        x0 = col * (W + border) + border
        sheet[y0 : y0 + H, x0 : x0 + W] = img
        cv2.putText(
            sheet, label, (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA,
        )

    save_png(sheet, path, overwrite=overwrite)
    logger.debug("Contact sheet saved: %s", path)


# ===========================================================================
# Histogram plot
# ===========================================================================

def save_histogram_plot(
    images: list[FloatImg],
    labels: list[str],
    mask: BoolMask | None,
    path: str | Path,
    *,
    nbins: int = 64,
    overwrite: bool = False,
) -> None:
    """Plot foreground histograms for the image set."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for img, label in zip(images, labels):
        px = img[mask].ravel() if mask is not None else img.ravel()
        h, edges = np.histogram(px, bins=nbins, range=(0.0, 255.0))
        centres = (edges[:-1] + edges[1:]) / 2
        ax.plot(centres, h, label=label)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    ax.set_title("Foreground histograms (post-SHINE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    logger.debug("Histogram plot saved: %s", path)


# ===========================================================================
# Spectrum plot
# ===========================================================================

def save_spectrum_plot(
    images: list[FloatImg],
    labels: list[str],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Plot rotational-average power spectra for the image set."""
    import matplotlib.pyplot as plt
    from shine import radial_spectrum

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for img, label in zip(images, labels):
        radii, power = radial_spectrum(img)
        ax.semilogy(radii[1:], power[1:], label=label)

    ax.set_xlabel("Spatial frequency (cycles/image)")
    ax.set_ylabel("Mean power")
    ax.set_title("Rotational-average power spectra (post-SHINE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    logger.debug("Spectrum plot saved: %s", path)


# ===========================================================================
# QC CSV
# ===========================================================================

def write_qc_csv(
    images: list[FloatImg],
    labels: list[str],
    mask: BoolMask | None,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write per-image statistics to a CSV file."""
    from shine import imstats, _histogram_rmse, _spectral_rmse

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stats = imstats(images, mask=mask)
    hist_rmse = _histogram_rmse(images, mask=mask)
    spec_rmse = _spectral_rmse(images, mask=mask)

    fieldnames = ["label", "mean", "std", "rms_contrast", "min", "max", "median",
                  "hist_rmse_vs_avg", "spec_rmse_vs_avg"]

    rows = []
    for label, s in zip(labels, stats):
        row = {"label": label}
        row.update(s)
        row["hist_rmse_vs_avg"] = f"{hist_rmse:.6f}"
        row["spec_rmse_vs_avg"] = f"{spec_rmse:.6f}"
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.debug("QC CSV saved: %s", path)
