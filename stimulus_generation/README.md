# Face Stimuli Generation Pipeline

Reproducible, no-GUI Python pipeline replicating the Dor-Ziderman et al. vMMR
death-denial paradigm stimulus set.  Replaces FantaMorph 4 (morphing) and the
MATLAB SHINE toolbox (image equalization) with an open, testable implementation.

---

## Output

Per participant, six 425 × 405 grayscale PNGs:

| File | Description |
|---|---|
| `self.png` | SHINE-equalized self face |
| `other.png` | SHINE-equalized other face |
| `morph50.png` | 50-50 morph of self and other, equalized |
| `self_target.png` | Self + opaque black sunglasses |
| `other_target.png` | Other + opaque black sunglasses |
| `morph50_target.png` | Morph + opaque black sunglasses |

Plus QC outputs (contact sheet, histograms, spectra, `qc_report.csv`,
`validation_report.txt`) and `processing_metadata.json`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the optional heavy FAN fallback detector (not needed by default):

```bash
pip install face-alignment torch torchvision
```

---

## Quick start

```bash
python stimulus_generation/create_face_stimuli.py \
    --subject sub-001 \
    --self    raw_faces/sub-001/self_raw.jpg \
    --other   raw_faces/sub-001/other_raw.jpg \
    --out     stimuli/faces/sub-001 \
    --save-intermediates
```

---

## GUI (Streamlit)

Launch the GUI:

```bash
streamlit run stimulus_generation/gui/app.py
```

Notes:

- Single-subject mode supports file uploads or local file paths.
- Batch mode expects a folder structure like:

```text
raw_faces/
  sub-001/
    self.jpg
    other.jpg
  sub-002/
    self.jpg
    other.jpg
```

- The app writes outputs under the base output directory, creating one
  subfolder per subject.
- Advanced settings start from config.yaml defaults and can be tuned in the UI.

For Streamlit Cloud deployment and a repo-level overview, see the root
README.md.

---

## Streamlit Cloud deployment

If you want to host the GUI publicly:

1. Push the repository to GitHub.
2. Create a new Streamlit Cloud app for the repo.
3. Set the **Main file path** to:

```text
stimulus_generation/gui/app.py
```

4. Use the root requirements.txt for dependencies.

---

## Data privacy

- Local use: images are processed on your machine only.
- Streamlit Cloud: uploaded images are processed on the Streamlit Cloud VM.

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--subject` | *(required)* | Participant ID |
| `--self` | *(required)* | Path to self face image |
| `--other` | *(required)* | Path to other face image |
| `--out` | *(required)* | Output directory |
| `--landmark-method` | `mediapipe` | `mediapipe` or `fan` |
| `--spectrum` | `sf` | `sf` (recommended) or `spec` |
| `--iterations` | `10` | SHINE iterations |
| `--match-population` | `triplet` | `triplet` (per-participant) or `set` |
| `--manual-landmarks` | off | Load landmarks from existing JSON |
| `--save-intermediates` | off | Save pre-SHINE aligned images |
| `--overwrite` | off | Overwrite existing outputs |
| `--qc-only` | off | Regenerate QC only (stimuli must exist) |

All flags override `config.yaml`.

---

## Module reference

### `shine.py` — SHINE port (highest priority)

Faithful Python port of Willenbockel et al. (2010).  All functions operate
on **float64** arrays with pixel values in **[0, 255]**.

| Function | Purpose |
|---|---|
| `lum_match` | Z-score luminance normalisation (mean/std only) |
| `avg_hist` | Average per-image histograms |
| `build_target_sorted` | Rank-averaged sorted target vector |
| `exact_hist_match` | Exact histogram specification with reproducible tie-breaking |
| `spec_match` | Full amplitude-spectrum match (SHINE specMatch) |
| `sf_match` | Rotational-average spectrum match (SHINE sfMatch, **recommended**) |
| `rescale` | Global linear rescale across the whole image set |
| `shine` | Main driver: iterative hist + spectrum equalization |
| `rmse`, `ssim_score` | QC metrics |
| `radial_spectrum` | Rotational-average power spectrum |
| `sf_plot`, `spectrum_plot` | Diagnostic plots |
| `imstats` | Per-image foreground statistics |

#### SHINE configuration knobs

All exposed in `config.yaml` under `shine:`:

| Key | Default | Notes |
|---|---|---|
| `spectrum` | `sf` | `sf` for faces (sfMatch), `spec` for full-amplitude (specMatch) |
| `iterations` | `10` | Max SHINE iterations; early-stop when both RMSEs converge |
| `tolerance` | `1e-4` | Early-stop delta threshold |
| `rescale_mode` | `all_in_range` | `all_in_range` (SHINE default) or `avg_clip` |
| `match_population` | `triplet` | `triplet` = per-participant {self, other, morph}; `set` = all participants |
| `do_hist` | `true` | Run exact histogram matching each iteration |

#### Important: global vs per-image rescale

`rescale()` applies **one** linear transform across the entire image set.
Per-image rescaling is forbidden — it would destroy the histogram and spectral
match already achieved.

#### Note on gamma

SHINE assumes a linearised (gamma-corrected) display. The saved PNGs contain
**linear** pixel values. Gamma correction for the physical monitor should be
applied at the PsychoPy display stage, not in the stimulus files.

---

### `align.py` — Similarity alignment

Uses `cv2.estimateAffinePartial2D` (4-DOF: rotation + uniform scale +
translation).  `cv2.getAffineTransform` (6-DOF, includes shear) is
**deliberately not used** for alignment — shear would distort face geometry.
The same transform is applied to both the image and the landmark array.

---

### `morph.py` — Delaunay morphing

- Appends 8 boundary control points (4 corners + 4 edge midpoints, identical
  in both images) before triangulation, ensuring full-frame coverage.
- Per-triangle affine warps (`cv2.getAffineTransform`) are correct at the
  intra-triangle level (this is distinct from the global similarity-only
  alignment step).
- Blends: `morph = (1-α)*warp_self + α*warp_other`.

---

### `landmarks.py`

- **Primary**: MediaPipe Face Mesh (478 points, no PyTorch dependency).
- **Fallback**: face-alignment FAN (68 points, requires torch).
- Validates exactly one face per image; fails loudly otherwise.
- `--manual-landmarks` loads corrected landmark JSON as the canonical artifact.

---

### `masks.py`

`create_face_mask(size, height_fraction, width_fraction)` → boolean oval mask.
The **same** mask is used for self / other / morph throughout, so the mask
edge is identical in every image.  SHINE re-applies the mask after each
spectral step to zero the background.

---

### `sunglasses.py`

Draws opaque black sunglasses on the **already-equalized** images.
SHINE is **not** run on target images (targets are only required to be
visually detectable, not SHINE-equalized).

---

### `validate.py`

(a) **Property tests** (always run): histogram RMSE, spectral RMSE, dtype,
    range, and background-zero checks.
(b) **Reference comparison** (if `reference_stimuli/` is populated): SSIM,
    histogram RMSE, and spectral RMSE vs original Dor-Ziderman stimuli.

---

## Running tests

```bash
cd stimulus_generation
python -m pytest tests/ -v
```

Individual suites:

```bash
python -m pytest tests/test_shine.py -v
python -m pytest tests/test_align.py -v
python -m pytest tests/test_morph.py -v
```

---

## Repo layout

```
stimulus_generation/
├── create_face_stimuli.py   # CLI entry point
├── config.yaml              # defaults (overridden by CLI flags)
├── requirements.txt
├── README.md
├── src/
│   ├── io_utils.py
│   ├── landmarks.py
│   ├── align.py
│   ├── morph.py
│   ├── masks.py
│   ├── shine.py
│   ├── sunglasses.py
│   ├── qc.py
│   └── validate.py
├── tests/
│   ├── test_shine.py
│   ├── test_align.py
│   └── test_morph.py
├── raw_faces/               # place input JPEGs here
├── outputs/                 # generated outputs
└── reference_stimuli/       # optional: original stimuli for validation
```

---

## Pipeline order

```
1  Load raw self/other (colour)
2  Detect landmarks (colour); save JSON + QC overlays
3  Similarity-align both to 425×405 template; transform landmarks
4  Convert to grayscale float64
5  Morph → morph50 (pre-mask, pre-equalization); boundary points added
6  Build common oval boolean mask
7  SHINE-equalize {self, other, morph} together within mask
8  Apply mask as final clamp (background = 0)
9  Draw sunglasses on equalized images → *_target.png
10 Save six PNGs + intermediates + processing_metadata.json
11 QC: contact sheet, overlays, histograms, spectra, qc_report.csv; validate
```

---

## References

- Willenbockel, V., Sadr, J., Fiset, D., Horne, G. O., Gosselin, F., &
  Tanaka, J. W. (2010). Controlling low-level image properties: The SHINE
  toolbox. *Behavior Research Methods*, 42(3), 671–684.
- Dor-Ziderman, Y., et al. (2019). Prediction-based neural mechanisms for
  shielding the self from existential threat. *NeuroImage*, 202, 116080.
