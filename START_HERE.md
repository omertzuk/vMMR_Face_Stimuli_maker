# Start Here

This repository generates face stimuli for the vMMR death-denial paradigm. It
provides a command-line pipeline, a Streamlit GUI, reusable image-processing
modules, and tests for the core operations.

## Repository structure

```text
.
├── README.md
├── START_HERE.md
├── requirements.txt
├── packages.txt
├── stimuli/
│   ├── Screenshot ...png          # input face images
│   └── faces/sub-001/             # generated participant output
└── stimulus_generation/
    ├── create_face_stimuli.py     # main six-stimulus pipeline
    ├── create_morph_sequence.py   # optional morph progression/video tool
    ├── config.yaml                # pipeline defaults
    ├── requirements.txt            # Python dependencies
    ├── gui/app.py                 # Streamlit interface
    ├── src/                       # alignment, morphing, SHINE, QC, etc.
    └── tests/                     # automated tests
```

### Important directories and files

- `stimuli/` contains raw input images and is also being used here for
  generated outputs. For larger projects, keep raw images and generated
  outputs in separate directories.
- `stimulus_generation/create_face_stimuli.py` is the main entry point for
  producing the six final images for one participant.
- `stimulus_generation/config.yaml` contains defaults such as output size,
  landmark detector, alignment geometry, mask shape, SHINE settings, and
  sunglasses dimensions.
- `stimulus_generation/src/` contains the implementation used by the entry
  points:
  - `landmarks.py`: MediaPipe detection and landmark overlays
  - `align.py`: similarity alignment
  - `morph.py`: Delaunay-based face morphing
  - `masks.py`: oval face mask and background clamping
  - `shine.py`: luminance and spatial-frequency equalization
  - `sunglasses.py`: target-mask drawing
  - `qc.py` and `validate.py`: diagnostics and validation
- `stimulus_generation/tests/` contains unit tests for alignment, morphing,
  SHINE, shine effects, and morph sequences.

## Installation

From the repository root, create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r stimulus_generation/requirements.txt
```

The root `requirements.txt` currently mirrors the stimulus-generation
dependencies and is also suitable when deploying the Streamlit app.

## Generate the six face stimuli

The CLI requires one self image, one other image, a participant identifier,
and an output directory. With the two images currently in this repository,
run:

```bash
.venv/bin/python stimulus_generation/create_face_stimuli.py \
  --subject sub-001 \
  --self "stimuli/Screenshot 2026-11-09 at 13.37.22.png" \
  --other "stimuli/Screenshot 2026-11-09 at 13.37.38.png" \
  --out "stimuli/faces/sub-001" \
  --overwrite
```

Alternatively, activate `.venv` first and replace `.venv/bin/python` with
`python`.

For another participant, use a separate output directory:

```bash
python stimulus_generation/create_face_stimuli.py \
  --subject sub-002 \
  --self raw_faces/sub-002/self.jpg \
  --other raw_faces/sub-002/other.jpg \
  --out stimuli/faces/sub-002
```

The default detector is MediaPipe. Add `--save-intermediates` to retain
aligned and pre-SHINE images. Add `--overwrite` when an output directory
already contains files that should be replaced.

## Command-line options

The most useful options are:

| Option | Purpose |
|---|---|
| `--subject ID` | Participant identifier stored in metadata |
| `--self PATH` | Self-face input image |
| `--other PATH` | Other-face input image |
| `--out PATH` | Output directory |
| `--landmark-method mediapipe\|fan` | Select landmark detector |
| `--morph-alpha VALUE` | Morph weight; `0` is self and `1` is other |
| `--spectrum sf\|spec` | SHINE spectrum-matching mode |
| `--iterations N` | Maximum SHINE iterations |
| `--match-population triplet\|set` | SHINE matching population |
| `--manual-landmarks` | Reuse landmark JSON files in the output directory |
| `--save-intermediates` | Save aligned and pre-SHINE images |
| `--overwrite` | Replace existing generated files |
| `--qc-only` | Rebuild QC outputs from existing final stimuli |

All options override corresponding values in
`stimulus_generation/config.yaml`. Run
`python stimulus_generation/create_face_stimuli.py --help` for the complete
argument list.

## Generated output

For `--out stimuli/faces/sub-001`, the main files are written under
`stimuli/faces/sub-001/stimuli/`:

| File | Contents |
|---|---|
| `self.png` | Equalized self face |
| `other.png` | Equalized other face |
| `morph50.png` | Equalized morph, normally a 50/50 blend |
| `self_target.png` | Self face with opaque black sunglasses |
| `other_target.png` | Other face with opaque black sunglasses |
| `morph50_target.png` | Morph with opaque black sunglasses |

Each final image is a `425 × 405` grayscale PNG by default. The parent output
directory also contains:

```text
stimuli/faces/sub-001/
├── landmarks/
│   ├── landmarks_self.json
│   ├── landmarks_other.json
│   ├── overlay_self.png
│   └── overlay_other.png
├── qc/
│   ├── contact_sheet.png
│   ├── histograms.png
│   ├── spectra.png
│   ├── qc_report.csv
│   └── validation_report.txt
├── stimuli/
│   └── six final stimulus PNGs
└── processing_metadata.json
```

When `--save-intermediates` is enabled, an additional `intermediates/`
directory contains aligned images and the pre-SHINE morph.

## Pipeline implemented by `create_face_stimuli.py`

The script loads defaults from `config.yaml`, then lets command-line flags
override them. It executes the following stages:

1. **Prepare the output directory.** The output directory is created if
   necessary. Existing files are protected unless `--overwrite` is supplied.
2. **Load the two colour images.** The self and other images are read from the
   paths passed through `--self` and `--other`.
3. **Detect or load facial landmarks.** MediaPipe Face Mesh is used by
   default. The detected points and visual overlays are saved in `landmarks/`.
   With `--manual-landmarks`, the existing JSON landmark files are loaded
   instead.
4. **Similarity-align the faces.** A rotation/scale/translation transform
   places both faces into the common `425 × 405` template. The same transform
   is applied to their landmarks, preserving corresponding facial geometry.
5. **Convert to grayscale.** Aligned colour images become floating-point
   luminance arrays in the range expected by the image-processing modules.
6. **Create the morph.** Delaunay-based triangle warping brings both aligned
   faces into a shared geometry and blends them using `morph.alpha` (default
   `0.5`).
7. **Build the face mask.** A common oval mask defines the foreground face
   region. Pixels outside the face are ultimately set to black.
8. **Run SHINE equalization.** The self, other, and morph images are processed
   together. Histogram matching and spatial-frequency matching reduce
   luminance and low-level image differences while retaining facial content.
   The default mode is rotational-average spatial-frequency matching (`sf`)
   with up to 10 iterations.
9. **Clamp the background.** The mask is applied to the equalized images so
   the surrounding background is black.
10. **Create target stimuli.** Opaque black sunglasses are drawn over the eye
    regions using the aligned landmarks. These produce the three
    `_target.png` files.
11. **Save metadata and quality-control outputs.** The script writes the six
    PNGs, processing settings, contact sheet, histograms, spectra, CSV
    statistics, and a validation report.

## Regenerating QC only

If the final stimuli already exist and only QC files need to be rebuilt:

```bash
.venv/bin/python stimulus_generation/create_face_stimuli.py \
  --subject sub-001 \
  --self unused \
  --other unused \
  --out stimuli/faces/sub-001 \
  --qc-only
```

In QC-only mode, the image paths and subject value are not used to process
images; the existing `stimuli/self.png`, `other.png`, and `morph50.png` files
are loaded from the output directory.

## Streamlit GUI

Launch the GUI from the repository root:

```bash
source .venv/bin/activate
streamlit run stimulus_generation/gui/app.py
```

The GUI supports single-subject uploads or local paths and batch processing.
Its advanced settings are based on `config.yaml`. For deployment instructions,
see the Streamlit section in `README.md`.

## Optional morph progression

`create_morph_sequence.py` is separate from the six-stimulus pipeline. It
creates intermediate morph images, optionally including a video, without the
SHINE equalization and sunglasses stages:

```bash
python stimulus_generation/create_morph_sequence.py \
  --self raw_faces/sub-001/self.jpg \
  --other raw_faces/sub-001/other.jpg \
  --out stimuli/morph_sequences/sub-001 \
  --num-intermediates 5 \
  --video \
  --video-duration 3.0
```

For `N` intermediates, the alpha values are
`1/(N+1), 2/(N+1), ..., N/(N+1)`, excluding the self and other endpoints.

## Testing

Run the test suite from the repository root after activating `.venv`:

```bash
python -m unittest discover -s stimulus_generation/tests -p "test_*.py"
```

The tests cover the numerical and geometric building blocks. When changing
configuration or pipeline behavior, also run the CLI on representative face
images and inspect the generated contact sheet and validation report.

## Data and privacy

The CLI processes images locally. Do not commit sensitive participant images or
generated outputs unless that is intentional for the project. The Streamlit
deployment processes uploaded images on the hosting machine rather than in the
browser.
