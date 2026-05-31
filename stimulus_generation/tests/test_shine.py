"""Property-based tests for shine.py.

Run with:
    cd stimulus_generation
    python -m pytest tests/test_shine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shine as sh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
H, W = 100, 100


def _random_images(n=3, h=H, w=W, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.uniform(10.0, 245.0, size=(h, w)) for _ in range(n)]


def _oval_mask(h=H, w=W):
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    return ((y - cy) / (h * 0.45)) ** 2 + ((x - cx) / (w * 0.40)) ** 2 <= 1.0


# ---------------------------------------------------------------------------
# lum_match
# ---------------------------------------------------------------------------

class TestLumMatch:
    def test_output_count(self):
        imgs = _random_images(3)
        out = sh.lum_match(imgs)
        assert len(out) == 3

    def test_shape_preserved(self):
        imgs = _random_images(3)
        out = sh.lum_match(imgs)
        for orig, result in zip(imgs, out):
            assert orig.shape == result.shape

    def test_mean_std_equalised(self):
        imgs = _random_images(3)
        out = sh.lum_match(imgs)
        means = [np.mean(o) for o in out]
        stds = [np.std(o) for o in out]
        assert np.allclose(means, np.mean(means), atol=1e-8)
        assert np.allclose(stds, np.mean(stds), atol=1e-8)

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        out = sh.lum_match(imgs, mask=mask)
        for o in out:
            assert np.all(o[~mask] == 0.0)


# ---------------------------------------------------------------------------
# build_target_sorted
# ---------------------------------------------------------------------------

class TestBuildTargetSorted:
    def test_length(self):
        mask = _oval_mask()
        n_fg = int(mask.sum())
        imgs = _random_images(3)
        t = sh.build_target_sorted(imgs, mask=mask)
        assert len(t) == n_fg

    def test_sorted(self):
        imgs = _random_images(3)
        t = sh.build_target_sorted(imgs)
        assert np.all(np.diff(t) >= 0)

    def test_single_image_identity(self):
        imgs = _random_images(1)
        t = sh.build_target_sorted(imgs)
        assert np.allclose(np.sort(imgs[0].ravel()), t)


# ---------------------------------------------------------------------------
# exact_hist_match
# ---------------------------------------------------------------------------

class TestExactHistMatch:
    def test_histogram_equals_target(self):
        imgs = _random_images(3)
        target = sh.build_target_sorted(imgs)
        rng = np.random.default_rng(7)
        out = sh.exact_hist_match(imgs[0], target, rng=rng)
        # After matching, sorted values should match target
        assert np.allclose(np.sort(out.ravel()), target, atol=1e-10)

    def test_shape_preserved(self):
        imgs = _random_images(3)
        target = sh.build_target_sorted(imgs)
        out = sh.exact_hist_match(imgs[0], target)
        assert out.shape == imgs[0].shape

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        target = sh.build_target_sorted(imgs, mask=mask)
        out = sh.exact_hist_match(imgs[0], target, mask=mask, rng=np.random.default_rng(1))
        assert np.all(out[~mask] == 0.0)

    def test_reproducible_with_seed(self):
        imgs = _random_images(3)
        target = sh.build_target_sorted(imgs)
        out1 = sh.exact_hist_match(imgs[0], target, rng=np.random.default_rng(42))
        out2 = sh.exact_hist_match(imgs[0], target, rng=np.random.default_rng(42))
        assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# spec_match
# ---------------------------------------------------------------------------

class TestSpecMatch:
    def test_output_count(self):
        imgs = _random_images(3)
        out = sh.spec_match(imgs)
        assert len(out) == 3

    def test_shape_preserved(self):
        imgs = _random_images(3)
        out = sh.spec_match(imgs)
        for orig, o in zip(imgs, out):
            assert orig.shape == o.shape

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        out = sh.spec_match(imgs, mask=mask)
        for o in out:
            assert np.all(o[~mask] == 0.0)


# ---------------------------------------------------------------------------
# sf_match
# ---------------------------------------------------------------------------

class TestSfMatch:
    def test_output_count(self):
        imgs = _random_images(3)
        out = sh.sf_match(imgs)
        assert len(out) == 3

    def test_shape_preserved(self):
        imgs = _random_images(3)
        out = sh.sf_match(imgs)
        for orig, o in zip(imgs, out):
            assert orig.shape == o.shape

    def test_radial_spectra_close_after_match(self):
        """After sf_match, radial spectra should be much more similar."""
        rng = np.random.default_rng(99)
        # Create images with deliberately different spectra
        imgs = [rng.uniform(0, 50 * (i + 1), (H, W)) for i in range(3)]
        before_rmse = sh._spectral_rmse(imgs, mask=None)
        out = sh.sf_match(imgs)
        after_rmse = sh._spectral_rmse(out, mask=None)
        assert after_rmse < before_rmse, (
            f"sf_match did not reduce spectral RMSE: {before_rmse:.4f} -> {after_rmse:.4f}"
        )

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        out = sh.sf_match(imgs, mask=mask)
        for o in out:
            assert np.all(o[~mask] == 0.0)


# ---------------------------------------------------------------------------
# rescale
# ---------------------------------------------------------------------------

class TestRescale:
    def test_global_range_all_in_range(self):
        imgs = _random_images(3)
        out = sh.rescale(imgs, mode="all_in_range")
        all_vals = np.concatenate([o.ravel() for o in out])
        assert float(all_vals.min()) >= 0.0 - 1e-8
        assert float(all_vals.max()) <= 255.0 + 1e-8

    def test_single_linear_transform(self):
        """Verify ONE linear map: a*(x-b) for constants a,b across all images."""
        imgs = _random_images(3)
        out = sh.rescale(imgs, mode="all_in_range")
        all_in = np.concatenate([i.ravel() for i in imgs])
        all_out = np.concatenate([o.ravel() for o in out])
        # Fit a linear regression; R² should be ~1
        coeffs = np.polyfit(all_in, all_out, 1)
        residuals = all_out - np.polyval(coeffs, all_in)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((all_out - all_out.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        assert r2 > 0.9999, f"Not a single linear transform: R²={r2}"

    def test_rank_order_preserved(self):
        """Global linear rescale must not change rank order."""
        imgs = _random_images(3)
        out = sh.rescale(imgs, mode="all_in_range")
        all_in = np.concatenate([i.ravel() for i in imgs])
        all_out = np.concatenate([o.ravel() for o in out])
        assert np.array_equal(np.argsort(all_in), np.argsort(all_out))

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        out = sh.rescale(imgs, mask=mask, mode="all_in_range")
        for o in out:
            assert np.all(o[~mask] == 0.0)


# ---------------------------------------------------------------------------
# shine (integration)
# ---------------------------------------------------------------------------

class TestShine:
    def test_output_shape_and_count(self):
        imgs = _random_images(3)
        out = sh.shine(imgs, iterations=2, rng=np.random.default_rng(0))
        assert len(out) == 3
        for orig, o in zip(imgs, out):
            assert orig.shape == o.shape

    def test_output_dtype_float64(self):
        imgs = _random_images(3)
        out = sh.shine(imgs, iterations=2, rng=np.random.default_rng(0))
        for o in out:
            assert o.dtype == np.float64

    def test_output_range(self):
        imgs = _random_images(3)
        out = sh.shine(imgs, iterations=2, rng=np.random.default_rng(0))
        all_vals = np.concatenate([o.ravel() for o in out])
        assert float(all_vals.min()) >= -1e-6
        assert float(all_vals.max()) <= 255.0 + 1e-6

    def test_histograms_closer_after_shine(self):
        imgs = _random_images(3, seed=77)
        before = sh._histogram_rmse(imgs, mask=None)
        out = sh.shine(imgs, iterations=4, rng=np.random.default_rng(0))
        after = sh._histogram_rmse(out, mask=None)
        assert after < before, f"SHINE did not improve histogram RMSE: {before:.2f} -> {after:.2f}"

    def test_spectra_closer_after_shine(self):
        rng_data = np.random.default_rng(11)
        imgs = [rng_data.uniform(0, 50 * (i + 1), (H, W)) for i in range(3)]
        before = sh._spectral_rmse(imgs, mask=None)
        out = sh.shine(imgs, iterations=4, rng=np.random.default_rng(0))
        after = sh._spectral_rmse(out, mask=None)
        assert after < before, f"SHINE did not improve spectral RMSE: {before:.2f} -> {after:.2f}"

    def test_mask_background_zero(self):
        imgs = _random_images(3)
        mask = _oval_mask()
        out = sh.shine(imgs, mask=mask, iterations=2, rng=np.random.default_rng(0))
        for o in out:
            assert np.all(o[~mask] == 0.0), "Background pixels must be zero after SHINE"

    def test_foreground_histograms_identical_after_hist_step(self):
        """After shine with do_hist=True, foreground histograms should be very close."""
        imgs = _random_images(3, seed=55)
        mask = _oval_mask()
        out = sh.shine(imgs, mask=mask, iterations=5, rng=np.random.default_rng(0))
        nbins = 64
        hists = []
        for o in out:
            px = o[mask].ravel()
            h, _ = np.histogram(px, bins=nbins, range=(0.0, 255.0))
            hists.append(h.astype(np.float64))
        # All histograms should be very similar
        for i in range(len(hists)):
            for j in range(i + 1, len(hists)):
                err = sh.rmse(hists[i], hists[j])
                assert err < 50.0, (  # lenient: spectral step re-perturbs hist
                    f"Histograms {i} and {j} diverged: RMSE={err:.2f}"
                )


# ---------------------------------------------------------------------------
# imstats
# ---------------------------------------------------------------------------

class TestImstats:
    def test_returns_correct_keys(self):
        imgs = _random_images(2)
        stats = sh.imstats(imgs)
        assert len(stats) == 2
        for s in stats:
            for k in ("mean", "std", "rms_contrast", "min", "max", "median"):
                assert k in s

    def test_masked_stats_exclude_background(self):
        imgs = _random_images(2)
        mask = _oval_mask()
        # Set background to an extreme value
        for img in imgs:
            img[~mask] = 999.0
        stats = sh.imstats(imgs, mask=mask)
        for s in stats:
            assert s["max"] < 999.0, "Background should be excluded from stats"


# ---------------------------------------------------------------------------
# radial_spectrum
# ---------------------------------------------------------------------------

class TestRadialSpectrum:
    def test_output_lengths_match(self):
        img = _random_images(1)[0]
        radii, power = sh.radial_spectrum(img)
        assert len(radii) == len(power)

    def test_non_negative_power(self):
        img = _random_images(1)[0]
        _, power = sh.radial_spectrum(img)
        assert np.all(power >= 0.0)
