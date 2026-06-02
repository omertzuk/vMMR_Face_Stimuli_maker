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
