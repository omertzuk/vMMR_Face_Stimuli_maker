# Face Stimuli Generator

Face-stimuli generation pipeline and Streamlit GUI for producing vMMR face
stimuli (self, other, morph) with SHINE equalization and QC outputs.

Maintained by Omer Tzuk, Contemplative Neuroscience and Neurophenomenology Lab,
University of Haifa.

## Quick start (GUI)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run stimulus_generation/gui/app.py
```

## Quick start (CLI)

```bash
python stimulus_generation/create_face_stimuli.py \
    --subject sub-001 \
    --self    raw_faces/sub-001/self.jpg \
    --other   raw_faces/sub-001/other.jpg \
    --out     stimuli/faces/sub-001 \
    --save-intermediates
```

## Morph sequence CLI

Generate a sequence of intermediate morphs between two face images, with an
optional video export that plays the full sequence from self to other.

```bash
python stimulus_generation/create_morph_sequence.py \
    --self raw_faces/sub-001/self.jpg \
    --other raw_faces/sub-001/other.jpg \
    --out stimuli/morph_sequences/sub-001 \
    --num-intermediates 5 \
    --video \
    --video-duration 3.0
```

### What it creates

- `self.png`: aligned and masked self endpoint
- `other.png`: aligned and masked other endpoint
- `morph_001_alpha_0.167.png`, `morph_002_alpha_0.333.png`, ...:
  intermediate-only morph PNGs
- `morph_sequence.mp4` when `--video` is enabled

### Alpha convention

`--num-intermediates N` creates exactly `N` intermediate morphs and excludes
the endpoints from the morph PNG sequence. The alpha step is:

$$
\Delta \alpha = \frac{1}{N + 1}
$$

So the generated morph alphas are `1 / (N + 1)`, `2 / (N + 1)`, ...,
`N / (N + 1)`. For example, `--num-intermediates 3` produces alphas `0.25`,
`0.50`, and `0.75`.

### Video mode

When `--video` is enabled, the video frame order is `self.png`, then all
intermediate morphs, then `other.png`. Use `--video-duration` to control the
total playback time in seconds; the script converts that into a per-frame
display time automatically.

### Useful options

- `--landmark-method mediapipe|fan`: choose the landmark detector
- `--manual-landmarks`: reuse landmarks already saved in the output folder
- `--overwrite`: replace existing outputs
- `--video-name filename.mp4`: choose a custom video filename

## Batch folder convention

```
raw_faces/
  sub-001/
    self.jpg
    other.jpg
  sub-002/
    self.jpg
    other.jpg
```

The GUI batch mode scans subfolders and creates one output folder per subject
under the chosen output base directory.

## Streamlit Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the **Main file path** to:

```
stimulus_generation/gui/app.py
```

4. Keep the default branch and select the root `requirements.txt`.

## Data privacy

- Local use: images are processed on your machine only.
- Streamlit Cloud: uploaded images are processed on the Streamlit Cloud VM.

## Documentation

- Full pipeline documentation: stimulus_generation/README.md

## License

MIT. See LICENSE.
