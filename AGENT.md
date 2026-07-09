# Agent Notes

This repository builds vMMR face stimuli from self/other face images. It has a
CLI pipeline and a Streamlit GUI wrapper. Future agents should treat this as an
experimental stimulus-generation codebase: small image-processing changes can
change the scientific output, so preserve the pipeline contracts unless the user
explicitly asks to revise them.

## Project Map

- `README.md` is the repo-level quick start and deployment overview.
- `stimulus_generation/README.md` is the detailed pipeline documentation.
- `stimulus_generation/create_face_stimuli.py` is the CLI entry point and main
  orchestration layer.
- `stimulus_generation/gui/app.py` is the Streamlit app. It imports
  `load_config` and `run_pipeline` from the CLI module.
- `stimulus_generation/config.yaml` contains defaults used by both CLI and GUI.
- `stimulus_generation/src/` contains the implementation modules:
  - `landmarks.py`: MediaPipe primary detector, optional FAN fallback.
  - `align.py`: similarity alignment to the output template.
  - `morph.py`: Delaunay face morphing.
  - `masks.py`: shared oval mask creation/application.
  - `shine.py`: Python SHINE implementation.
  - `sunglasses.py`: target stimulus overlay.
  - `qc.py` and `validate.py`: QC plots, metrics, and validation reports.
- `stimulus_generation/tests/` contains focused tests for SHINE, alignment, and
  morphing behavior.

## Setup And Commands

Create an environment and install the root requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the GUI:

```bash
streamlit run stimulus_generation/gui/app.py
```

Run the CLI for one subject:

```bash
python stimulus_generation/create_face_stimuli.py \
    --subject sub-001 \
    --self raw_faces/sub-001/self.jpg \
    --other raw_faces/sub-001/other.jpg \
    --out stimuli/faces/sub-001 \
    --save-intermediates
```

Run tests:

```bash
cd stimulus_generation
python -m pytest tests/ -v
```

## Pipeline Contract

The intended final output per subject is six grayscale PNGs at 425 x 405:

- `self.png`
- `other.png`
- `morph50.png`
- `self_target.png`
- `other_target.png`
- `morph50_target.png`

The output folder also includes `processing_metadata.json`, landmark JSON and
overlays, optional intermediates, and QC outputs under `qc/`.

The main pipeline order is:

1. Load self and other color images.
2. Detect or load landmarks.
3. Similarity-align both images and transform landmarks.
4. Convert aligned images to grayscale float images.
5. Build a 50/50 morph before equalization.
6. Create one common oval face mask.
7. SHINE-equalize self, other, and morph together within that mask.
8. Reapply the mask and zero the background.
9. Draw sunglasses on already equalized images to make target images.
10. Save stimuli, metadata, and QC outputs.

## Important Invariants

- SHINE functions operate on `float64` arrays with values in `[0, 255]`.
- Do not apply per-image rescaling after SHINE. The code intentionally uses a
  global transform across the image set.
- Keep the same mask for self, other, and morph so the mask edge is identical.
- Sunglasses are drawn after SHINE. Do not SHINE-equalize target images unless
  the scientific requirement changes.
- Alignment should stay similarity-only: rotation, translation, and uniform
  scale. Avoid shear in the global alignment step.
- Gamma correction belongs in the display/experiment stage, not in saved PNGs.
- Landmark detection should fail loudly if there is not exactly one face.
- Manual landmarks are loaded from existing landmark JSON files and should be
  treated as canonical corrected artifacts.

## GUI Notes

`stimulus_generation/gui/app.py` resolves relative paths from the repo root and
writes one output folder per subject under the selected base output directory.
Single-subject mode supports uploads or local file paths. Batch mode scans
subdirectories with configurable `self` and `other` filenames.

The GUI advanced settings mirror `config.yaml`. If you add config keys, update
both `config.yaml` and the GUI defaults/overrides so CLI and GUI behavior stay
aligned.

## Dependencies And Network

The default landmark method is MediaPipe. The repo includes
`stimulus_generation/src/face_landmarker.task`, but `landmarks.py` can download
the model if it is missing. Avoid introducing new runtime network requirements
without making them optional and documenting them.

The FAN fallback is optional and heavy. It requires `face-alignment`, `torch`,
and `torchvision`, which are intentionally commented out in the requirements.

## Working Practices

- Prefer changing the implementation module that owns the behavior, then keep
  `create_face_stimuli.py` as orchestration glue.
- Keep generated subject images, raw face photos, and large stimulus outputs out
  of commits unless the user explicitly asks to add fixtures or reference data.
- If changing image-processing behavior, update or add focused tests and run the
  relevant suite under `stimulus_generation/tests/`.
- If changing CLI flags or config defaults, update both READMEs and the GUI
  controls when applicable.
- Be careful with privacy: uploaded/local face images may identify subjects.
