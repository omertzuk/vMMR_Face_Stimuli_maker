"""
Faithful Python port of the MATLAB SHINE toolbox
(Willenbockel et al., 2010 — Behavior Research Methods).

All functions operate on float64 arrays with pixel values in [0, 255].
When a boolean `mask` is provided (True = foreground/face pixel), only
foreground pixels participate in statistics; background pixels are forced
to 0 in all outputs.

FFT convention used throughout
-------------------------------
    F = np.fft.fft2(img)
    A = np.abs(F)       # amplitude
    P = np.angle(F)     # phase
    reconstruct: np.real(np.fft.ifft2(A * np.exp(1j * P)))

fftshift is used ONLY for radial-bin indexing and plotting; ifftshift is
applied before ifft2 when the amplitude was manipulated in shifted space.

NOTE on gamma: SHINE assumes a linear (linearised) monitor. Gamma correction
belongs at the PsychoPy display stage, not in the saved PNGs. The pipeline
does not apply any gamma correction to the output files.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatImg = NDArray[np.float64]
BoolMask = NDArray[np.bool_]


# ===========================================================================
# 1. Luminance matching (z-score normalisation)
# ===========================================================================

def lum_match(
    images: list[FloatImg],
    mask: BoolMask | None = None,
) -> list[FloatImg]:
    """Match luminance statistics across images.

    For each image X:
        Z = (X - m) / s
        E = Z * S + M
    where m, s are that image's foreground mean/std and M, S are the
    averages of all images' means and stds (SHINE defaults).

    NOTE: exact_hist_match subsumes this — identical histograms imply
    identical mean and std — so lum_match is only for 'luminance-only' mode.
    """
    per_mean = []
    per_std = []
    for img in images:
        px = img[mask] if mask is not None else img.ravel()
        per_mean.append(float(np.mean(px)))
        per_std.append(float(np.std(px, ddof=0)))

    M = float(np.mean(per_mean))
    S = float(np.mean(per_std))

    out = []
    for img, m, s in zip(images, per_mean, per_std):
        if s == 0:
            z = np.zeros_like(img)
        else:
            z = (img - m) / s
        e = z * S + M
        if mask is not None:
            result = np.zeros_like(img)
            result[mask] = e[mask]
        else:
            result = e
        out.append(result)
    return out


# ===========================================================================
# 2. Average histogram
# ===========================================================================

def avg_hist(
    images: list[FloatImg],
    mask: BoolMask | None = None,
    nbins: int = 256,
) -> NDArray[np.float64]:
    """Return the per-image (masked) histograms averaged across images.

    Returns a 1-D float64 array of length nbins whose values are normalised
    counts (probability mass), averaged over the image set.
    """
    hist_sum = np.zeros(nbins, dtype=np.float64)
    for img in images:
        px = img[mask].ravel() if mask is not None else img.ravel()
        h, _ = np.histogram(px, bins=nbins, range=(0.0, 255.0))
        hist_sum += h.astype(np.float64)
    return hist_sum / len(images)


# ===========================================================================
# 3. Build sorted target vector for exact histogram matching
# ===========================================================================

def build_target_sorted(
    images: list[FloatImg],
    mask: BoolMask | None = None,
) -> NDArray[np.float64]:
    """Build the rank-averaged target value vector (SHINE tarhist).

    For each image, sort its foreground pixel values ascending, then average
    across images by rank. The result is a 1-D float64 array of length N
    (number of foreground pixels per image). All images must have the same
    mask (same N).
    """
    sorted_vals = []
    for img in images:
        px = img[mask].ravel() if mask is not None else img.ravel()
        sorted_vals.append(np.sort(px.astype(np.float64)))

    # Stack (n_images, N) and average over axis=0
    stacked = np.stack(sorted_vals, axis=0)
    return np.mean(stacked, axis=0)


# ===========================================================================
# 4. Exact histogram matching
# ===========================================================================

def exact_hist_match(
    image: FloatImg,
    target_sorted: NDArray[np.float64],
    mask: BoolMask | None = None,
    rng: np.random.Generator | None = None,
) -> FloatImg:
    """SHINE exact global histogram specification.

    Algorithm:
    1. Extract foreground pixels and sort by value ascending.
    2. Randomise ties using a seeded RNG for reproducibility.
    3. Assign target_sorted[k] to the k-th ranked source pixel.
    4. Write values back to foreground positions; background stays at 0.

    Parameters
    ----------
    image:
        Input float64 image.
    target_sorted:
        Sorted target values (length = number of foreground pixels).
    mask:
        Boolean mask (True = foreground). None means all pixels.
    rng:
        np.random.Generator for tie-breaking. None creates a default RNG.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    result = np.zeros_like(image)

    if mask is not None:
        indices = np.where(mask.ravel())[0]
        px = image.ravel()[indices]
    else:
        indices = np.arange(image.size)
        px = image.ravel()

    N = len(px)
    if N != len(target_sorted):
        raise ValueError(
            f"Pixel count mismatch: image has {N} foreground px, "
            f"target has {len(target_sorted)} values."
        )

    # Argsort with random tie-breaking: add tiny noise to break ties
    noise = rng.random(N) * 1e-10
    order = np.argsort(px + noise, kind="stable")

    # Map k-th ranked pixel → target_sorted[k]
    assigned = np.empty(N, dtype=np.float64)
    assigned[order] = target_sorted

    flat = result.ravel()
    flat[indices] = assigned
    result = flat.reshape(image.shape)
    return result


# ===========================================================================
# 5. Full amplitude-spectrum matching (SHINE specMatch)
# ===========================================================================

def spec_match(
    images: list[FloatImg],
    target_amp: FloatImg | None = None,
    mask: BoolMask | None = None,
) -> list[FloatImg]:
    """Replace each image's full amplitude spectrum with a common target.

    target_amp defaults to the mean |FFT2| across all images.
    If mask is given, background is zeroed AFTER the IFFT.

    NOTE: This produces cloudy-looking results; for faces use sf_match.
    """
    ffts = [np.fft.fft2(img) for img in images]
    amps = [np.abs(F) for F in ffts]

    if target_amp is None:
        target_amp = np.mean(np.stack(amps, axis=0), axis=0)

    out = []
    for F in ffts:
        phase = np.angle(F)
        reconstructed = np.real(np.fft.ifft2(target_amp * np.exp(1j * phase)))
        if mask is not None:
            reconstructed = np.where(mask, reconstructed, 0.0)
        out.append(reconstructed)
    return out


# ===========================================================================
# 6. Rotational-average spectrum matching (SHINE sfMatch) — RECOMMENDED
# ===========================================================================

def sf_match(
    images: list[FloatImg],
    target_amp: FloatImg | None = None,
    mask: BoolMask | None = None,
) -> list[FloatImg]:
    """Match each image's rotational-average (radial) amplitude spectrum.

    Preserves orientation content; produces higher-quality results than
    spec_match for face images (SHINE sfMatch).

    Algorithm
    ---------
    1. Compute the fftshifted amplitude for each image.
    2. Build a radius grid r = round(sqrt(fx² + fy²)) in shifted coords.
    3. target_amp defaults to mean |FFT2| (unshifted) across images.
    4. For each radius shell r:
           src_sum = sum of this image's amplitudes at shell r
           tgt_sum = sum of target amplitudes at shell r
           coeff_r = tgt_sum / src_sum   (guard src_sum == 0)
    5. Multiply all amplitudes in shell r by coeff_r.
    6. Unshift, recombine with original phase, IFFT, take real.
    7. Re-apply mask (bg = 0).
    """
    H, W = images[0].shape[:2]

    # Frequency axes in fftshifted coordinates
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=1.0)) * H   # cycles / image-height
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0)) * W
    FX, FY = np.meshgrid(fx, fy)
    radius_grid = np.round(np.sqrt(FX**2 + FY**2)).astype(int)
    max_r = int(radius_grid.max())

    # FFTs (unshifted)
    ffts = [np.fft.fft2(img) for img in images]
    amps_shifted = [np.fft.fftshift(np.abs(F)) for F in ffts]

    if target_amp is None:
        raw_amps = [np.abs(F) for F in ffts]
        target_amp_unshifted = np.mean(np.stack(raw_amps, axis=0), axis=0)
        target_amp_shifted = np.fft.fftshift(target_amp_unshifted)
    else:
        target_amp_shifted = np.fft.fftshift(target_amp)

    # Precompute per-shell target sums
    tgt_sums = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        shell = radius_grid == r
        tgt_sums[r] = target_amp_shifted[shell].sum()

    out = []
    for fft_img, amp_shifted in zip(ffts, amps_shifted):
        new_amp_shifted = amp_shifted.copy()

        for r in range(max_r + 1):
            shell = radius_grid == r
            src_sum = amp_shifted[shell].sum()
            if src_sum == 0.0:
                continue
            coeff = tgt_sums[r] / src_sum
            new_amp_shifted[shell] *= coeff

        # Convert back to unshifted amplitude and recombine with original phase
        new_amp = np.fft.ifftshift(new_amp_shifted)
        phase = np.angle(fft_img)
        reconstructed = np.real(np.fft.ifft2(new_amp * np.exp(1j * phase)))

        if mask is not None:
            reconstructed = np.where(mask, reconstructed, 0.0)
        out.append(reconstructed)
    return out


# ===========================================================================
# 7. Global rescale (CRITICAL: one linear transform across entire set)
# ===========================================================================

def rescale(
    images: list[FloatImg],
    mask: BoolMask | None = None,
    mode: str = "all_in_range",
) -> list[FloatImg]:
    """Rescale the entire image set with ONE global linear transform.

    Per-image rescaling is FORBIDDEN — it would break the spectral and
    histogram match already established.

    Modes
    -----
    "all_in_range" (SHINE default):
        global_min → 0, global_max → 255.
        All pixel values of all images land in [0, 255].

    "avg_clip":
        mean(per-image min) → 0, mean(per-image max) → 255.
        Allows some clipping but preserves more contrast on average.
    """
    if mode == "all_in_range":
        # Gather min/max across ALL images (foreground only if masked)
        all_vals = []
        for img in images:
            px = img[mask] if mask is not None else img.ravel()
            all_vals.append(px)
        concat = np.concatenate([v.ravel() for v in all_vals])
        global_min = float(concat.min())
        global_max = float(concat.max())

        if global_max == global_min:
            logger.warning("rescale: all pixels have the same value; returning zeros.")
            out = [np.zeros_like(img) for img in images]
        else:
            a = 255.0 / (global_max - global_min)
            out = [(img - global_min) * a for img in images]

    elif mode == "avg_clip":
        per_min = []
        per_max = []
        for img in images:
            px = img[mask] if mask is not None else img.ravel()
            per_min.append(float(px.min()))
            per_max.append(float(px.max()))
        g_min = float(np.mean(per_min))
        g_max = float(np.mean(per_max))
        if g_max == g_min:
            out = [np.zeros_like(img) for img in images]
        else:
            a = 255.0 / (g_max - g_min)
            out = [(img - g_min) * a for img in images]
    else:
        raise ValueError(f"Unknown rescale mode: {mode!r}")

    # Re-apply mask: background pixels must be 0
    if mask is not None:
        out = [np.where(mask, im, 0.0) for im in out]

    return out


# ===========================================================================
# 8. Main SHINE driver
# ===========================================================================

def shine(
    images: list[FloatImg],
    mask: BoolMask | None = None,
    do_hist: bool = True,
    spectrum: str = "sf",
    iterations: int = 10,
    order: tuple[str, str] = ("hist", "spec"),
    rescale_mode: str = "all_in_range",
    rng: np.random.Generator | None = None,
    tolerance: float = 1e-4,
) -> list[FloatImg]:
    """Faithful SHINE driver.

    One iteration = histogram match (optional) then spectral match,
    with the target RECOMPUTED each iteration (per Willenbockel Fig. 5).

    After each spectral step the mask is re-applied (bg = 0). This
    re-introduces the same oval edge in every image identically, so
    RELATIVE matching is preserved; the spectral match is therefore
    approximate. QC reports residual spectral RMSE for convergence
    monitoring.

    Greatest gains occur in the first ~4 iterations; the loop exits
    early when both spectral-RMSE and histogram-RMSE deltas fall below
    `tolerance`.

    For faces use spectrum="sf" (default).

    Parameters
    ----------
    images:   list of float64 arrays in [0, 255]
    mask:     boolean foreground mask (True = face pixel)
    do_hist:  if True, run exact histogram match each iteration
    spectrum: "sf" (sfMatch) | "spec" (specMatch)
    iterations: maximum number of iterations
    order:    ("hist","spec") — order within each iteration (SHINE default)
    rescale_mode: passed to rescale()
    rng:      seeded RNG for reproducible tie-breaking in hist match
    tolerance: early-stop threshold for delta in hist+spec RMSE
    """
    if rng is None:
        rng = np.random.default_rng(0)

    imgs = [img.copy().astype(np.float64) for img in images]
    spec_fn = sf_match if spectrum == "sf" else spec_match

    prev_hist_rmse = np.inf
    prev_spec_rmse = np.inf

    for it in range(1, iterations + 1):
        # --- Histogram step ---
        if do_hist and "hist" in order:
            target_sorted = build_target_sorted(imgs, mask=mask)
            imgs = [
                exact_hist_match(img, target_sorted, mask=mask, rng=rng)
                for img in imgs
            ]
            # Re-apply mask after hist step
            if mask is not None:
                imgs = [np.where(mask, im, 0.0) for im in imgs]

        # --- Spectral step ---
        if "spec" in order:
            imgs = spec_fn(imgs, target_amp=None, mask=mask)
            # Re-apply mask: spectrum step smears energy into background
            if mask is not None:
                imgs = [np.where(mask, im, 0.0) for im in imgs]

        # --- Convergence check ---
        hist_rmse = _histogram_rmse(imgs, mask)
        spec_rmse = _spectral_rmse(imgs, mask)

        d_hist = abs(prev_hist_rmse - hist_rmse)
        d_spec = abs(prev_spec_rmse - spec_rmse)

        logger.info(
            "  SHINE iter %d/%d  hist_RMSE=%.6f (Δ%.2e)  spec_RMSE=%.6f (Δ%.2e)",
            it, iterations, hist_rmse, d_hist, spec_rmse, d_spec,
        )

        if it > 1 and d_hist < tolerance and d_spec < tolerance:
            logger.info("  Early stop at iteration %d (both deltas < %.2e)", it, tolerance)
            break

        prev_hist_rmse = hist_rmse
        prev_spec_rmse = spec_rmse

    # --- Final global rescale ---
    imgs = rescale(imgs, mask=mask, mode=rescale_mode)
    return imgs


# ===========================================================================
# 9. QC helpers
# ===========================================================================

def rmse(a: FloatImg, b: FloatImg) -> float:
    """Root-mean-square error between two arrays of the same shape."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def ssim_score(a: FloatImg, b: FloatImg) -> float:
    """SSIM between two float64 images (values in [0, 255])."""
    from skimage.metrics import structural_similarity
    data_range = 255.0
    return float(structural_similarity(a, b, data_range=data_range))


def radial_spectrum(img: FloatImg) -> tuple[NDArray, NDArray]:
    """Compute the rotational-average power spectrum.

    Returns
    -------
    radii : 1-D int array of spatial frequencies (in cycles/image)
    power : 1-D float array of mean power at each radius
    """
    H, W = img.shape[:2]
    F = np.fft.fftshift(np.fft.fft2(img))
    power_map = np.abs(F) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(H)) * H
    fx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    FX, FY = np.meshgrid(fx, fy)
    r = np.round(np.sqrt(FX**2 + FY**2)).astype(int)

    max_r = int(r.max())
    radii = np.arange(0, max_r + 1)
    power = np.array(
        [power_map[r == rr].mean() if (r == rr).any() else 0.0 for rr in radii],
        dtype=np.float64,
    )
    return radii, power


def sf_plot(
    images: list[FloatImg],
    labels: list[str],
    path: str | None = None,
) -> None:
    """Plot rotational-average spectra for a set of images."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for img, label in zip(images, labels):
        radii, power = radial_spectrum(img)
        ax.semilogy(radii[1:], power[1:], label=label)
    ax.set_xlabel("Spatial frequency (cycles/image)")
    ax.set_ylabel("Mean power")
    ax.set_title("Rotational-average power spectra")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=120)
    else:
        plt.show()
    plt.close(fig)


def spectrum_plot(
    images: list[FloatImg],
    labels: list[str],
    path: str | None = None,
) -> None:
    """Plot 2-D log amplitude spectra side by side."""
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        F = np.fft.fftshift(np.fft.fft2(img))
        ax.imshow(np.log1p(np.abs(F)), cmap="inferno", origin="lower")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Log amplitude spectra")
    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=120)
    else:
        plt.show()
    plt.close(fig)


def imstats(
    images: list[FloatImg],
    mask: BoolMask | None = None,
) -> list[dict]:
    """Return masked per-image statistics: mean, std, rms-contrast, min, max, median."""
    stats = []
    for img in images:
        px = img[mask].ravel() if mask is not None else img.ravel()
        mean_val = float(np.mean(px))
        std_val = float(np.std(px, ddof=0))
        stats.append({
            "mean": mean_val,
            "std": std_val,
            "rms_contrast": std_val / mean_val if mean_val != 0 else 0.0,
            "min": float(px.min()),
            "max": float(px.max()),
            "median": float(np.median(px)),
        })
    return stats


# ===========================================================================
# 10. Advanced: Avanaki SSIM-gradient histogram optimisation (gated)
# ===========================================================================

def avanaki_ssim_hist_match(
    image: FloatImg,
    target_sorted: NDArray[np.float64],
    mask: BoolMask | None = None,
    iterations: int = 50,
    step_size: float = 0.1,
) -> FloatImg:
    """Experimental: Avanaki SSIM-gradient histogram optimisation.

    Preserves the exact target histogram while maximising SSIM to the
    original image via gradient ascent. Gate behind a flag — off by default.

    Reference: Avanaki (2009) "Exact global histogram specification
    optimized for structural similarity."
    """
    raise NotImplementedError(
        "Avanaki SSIM-gradient optimisation is not yet implemented. "
        "Pass advanced=False (default) to shine()."
    )


# ===========================================================================
# Internal helpers
# ===========================================================================

def _histogram_rmse(images: list[FloatImg], mask: BoolMask | None) -> float:
    """RMSE between per-image histograms and their average."""
    nbins = 256
    hists = []
    for img in images:
        px = img[mask].ravel() if mask is not None else img.ravel()
        h, _ = np.histogram(px, bins=nbins, range=(0.0, 255.0))
        hists.append(h.astype(np.float64))
    avg = np.mean(np.stack(hists), axis=0)
    errors = [rmse(h, avg) for h in hists]
    return float(np.mean(errors))


def _spectral_rmse(images: list[FloatImg], mask: BoolMask | None) -> float:
    """RMSE between per-image rotational-average spectra and their average."""
    spectra = []
    for img in images:
        _, power = radial_spectrum(img)
        spectra.append(power)

    # All spectra have the same length (same image size)
    arr = np.stack(spectra, axis=0)
    avg = np.mean(arr, axis=0)
    errors = [rmse(s, avg) for s in spectra]
    return float(np.mean(errors))
