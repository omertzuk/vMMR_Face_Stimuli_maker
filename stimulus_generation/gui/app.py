from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_TITLE = "Face Stimuli Generator"

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# ---------------------------------------------------------------------------
# Ensure stimulus_generation/ is importable
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from create_face_stimuli import load_config, run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            target.setdefault(key, {})
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_upload_dir() -> Path:
    if "upload_dir" not in st.session_state:
        temp_root = Path(tempfile.mkdtemp(prefix="stimuli_uploads_"))
        st.session_state["upload_dir"] = str(temp_root)
    return Path(st.session_state["upload_dir"])


def save_upload(upload, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as fh:
        fh.write(upload.getbuffer())


def build_cfg(overrides: dict, pipeline_flags: dict) -> dict:
    cfg = load_config()
    deep_update(cfg, overrides)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["save_intermediates"] = pipeline_flags["save_intermediates"]
    cfg["pipeline"]["overwrite"] = pipeline_flags["overwrite"]
    cfg["pipeline"]["qc_only"] = pipeline_flags["qc_only"]
    return cfg


def run_subject(
    subject_id: str,
    self_path: Path,
    other_path: Path,
    out_base: Path,
    overrides: dict,
    pipeline_flags: dict,
) -> Path:
    out_dir = out_base / subject_id
    args = argparse.Namespace(
        subject=subject_id,
        self_img=str(self_path),
        other_img=str(other_path),
        out=str(out_dir),
        manual_landmarks=pipeline_flags["manual_landmarks"],
    )
    cfg = build_cfg(overrides, pipeline_flags)
    run_pipeline(args, cfg)
    return out_dir


def scan_batch(root_dir: Path, self_name: str, other_name: str) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    if not root_dir.exists():
        return subjects
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        self_path = child / self_name
        other_path = child / other_name
        subjects.append(
            {
                "subject": child.name,
                "self_path": self_path,
                "other_path": other_path,
                "valid": self_path.exists() and other_path.exists(),
            }
        )
    return subjects


def show_qc(out_dir: Path) -> None:
    qc_dir = out_dir / "qc"
    if not qc_dir.exists():
        return

    contact = qc_dir / "contact_sheet.png"
    hist = qc_dir / "histograms.png"
    spec = qc_dir / "spectra.png"
    report = qc_dir / "validation_report.txt"

    if contact.exists():
        st.image(str(contact), caption="Contact sheet", use_container_width=True)
    if hist.exists():
        st.image(str(hist), caption="Histograms", use_container_width=True)
    if spec.exists():
        st.image(str(spec), caption="Spectra", use_container_width=True)
    if report.exists():
        st.text(report.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.write("Generate self/other/morph face stimuli with optional batch processing.")

cfg_defaults = load_config()
mask_defaults = cfg_defaults.get("mask", {})
shine_defaults = cfg_defaults.get("shine", {})
landmark_defaults = cfg_defaults.get("landmarks", {})
align_defaults = cfg_defaults.get("alignment", {})
sunglasses_defaults = cfg_defaults.get("sunglasses", {})

mode = st.radio("Mode", ["Single subject", "Batch folder"], horizontal=True)

out_base_str = st.text_input("Base output directory", value="stimuli/faces")
out_base = resolve_path(out_base_str)

with st.expander("Advanced settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mask")
        mask_width = st.number_input(
            "Width fraction",
            min_value=0.40,
            max_value=1.00,
            value=float(mask_defaults.get("width_fraction", 0.80)),
            step=0.01,
        )
        mask_height = st.number_input(
            "Height fraction",
            min_value=0.40,
            max_value=1.00,
            value=float(mask_defaults.get("height_fraction", 0.95)),
            step=0.01,
        )

        st.subheader("Landmarks")
        landmark_method = st.selectbox(
            "Method",
            ["mediapipe", "fan"],
            index=0 if landmark_defaults.get("method", "mediapipe") == "mediapipe" else 1,
        )
        manual_landmarks = st.checkbox("Use manual landmarks JSON", value=False)

    with col2:
        st.subheader("SHINE")
        spectrum = st.selectbox(
            "Spectrum",
            ["sf", "spec"],
            index=0 if shine_defaults.get("spectrum", "sf") == "sf" else 1,
        )
        iterations = st.number_input(
            "Iterations",
            min_value=1,
            max_value=100,
            value=int(shine_defaults.get("iterations", 10)),
            step=1,
        )
        tolerance = st.number_input(
            "Tolerance",
            min_value=1e-8,
            max_value=1e-2,
            value=float(shine_defaults.get("tolerance", 1e-4)),
            format="%.6f",
        )
        rescale_mode = st.selectbox(
            "Rescale mode",
            ["all_in_range", "avg_clip"],
            index=0 if shine_defaults.get("rescale_mode", "all_in_range") == "all_in_range" else 1,
        )
        match_population = st.selectbox(
            "Match population",
            ["triplet", "set"],
            index=0 if shine_defaults.get("match_population", "triplet") == "triplet" else 1,
        )
        do_hist = st.checkbox("Histogram match", value=bool(shine_defaults.get("do_hist", True)))

    st.subheader("Alignment")
    align_col1, align_col2 = st.columns(2)
    with align_col1:
        interocular = st.number_input(
            "Interocular distance (px)",
            min_value=50,
            max_value=300,
            value=int(align_defaults.get("interocular_distance", 140)),
            step=1,
        )
    with align_col2:
        eye_y = st.number_input(
            "Eye centre Y (px)",
            min_value=50,
            max_value=300,
            value=int(align_defaults.get("eye_centre_y", 160)),
            step=1,
        )

    st.subheader("Sunglasses")
    sun_col1, sun_col2, sun_col3 = st.columns(3)
    with sun_col1:
        lens_scale = st.number_input(
            "Lens scale",
            min_value=0.5,
            max_value=3.0,
            value=float(sunglasses_defaults.get("lens_scale", 1.6)),
            step=0.05,
        )
    with sun_col2:
        lens_height = st.number_input(
            "Lens height fraction",
            min_value=0.2,
            max_value=1.0,
            value=float(sunglasses_defaults.get("lens_height_frac", 0.55)),
            step=0.05,
        )
    with sun_col3:
        bridge_height = st.number_input(
            "Bridge height (px)",
            min_value=1,
            max_value=50,
            value=int(sunglasses_defaults.get("bridge_height", 6)),
            step=1,
        )

    st.subheader("Pipeline")
    pipe_col1, pipe_col2, pipe_col3 = st.columns(3)
    with pipe_col1:
        save_intermediates = st.checkbox("Save intermediates", value=False)
    with pipe_col2:
        overwrite = st.checkbox("Overwrite outputs", value=False)
    with pipe_col3:
        qc_only = st.checkbox("QC only", value=False)


overrides = {
    "landmarks": {"method": landmark_method},
    "mask": {"width_fraction": mask_width, "height_fraction": mask_height},
    "shine": {
        "spectrum": spectrum,
        "iterations": int(iterations),
        "tolerance": float(tolerance),
        "rescale_mode": rescale_mode,
        "match_population": match_population,
        "do_hist": bool(do_hist),
    },
    "alignment": {
        "interocular_distance": int(interocular),
        "eye_centre_y": int(eye_y),
    },
    "sunglasses": {
        "lens_scale": float(lens_scale),
        "lens_height_frac": float(lens_height),
        "bridge_height": int(bridge_height),
    },
}

pipeline_flags = {
    "save_intermediates": save_intermediates,
    "overwrite": overwrite,
    "qc_only": qc_only,
    "manual_landmarks": manual_landmarks,
}

if mode == "Single subject":
    st.subheader("Single subject")
    subject_id = st.text_input("Subject ID", value="sub-001")

    input_method = st.radio("Input method", ["Upload files", "Use file paths"], horizontal=True)

    if input_method == "Upload files":
        self_upload = st.file_uploader("Self image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        other_upload = st.file_uploader("Other image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if st.button("Run", type="primary"):
            if not subject_id:
                st.error("Subject ID is required.")
            elif self_upload is None or other_upload is None:
                st.error("Please upload both self and other images.")
            else:
                upload_dir = get_upload_dir() / subject_id
                self_path = upload_dir / f"self.{self_upload.name.split('.')[-1]}"
                other_path = upload_dir / f"other.{other_upload.name.split('.')[-1]}"
                save_upload(self_upload, self_path)
                save_upload(other_upload, other_path)
                with st.spinner("Processing..."):
                    try:
                        out_dir = run_subject(
                            subject_id,
                            self_path,
                            other_path,
                            out_base,
                            overrides,
                            pipeline_flags,
                        )
                        st.success(f"Done. Output: {out_dir}")
                        show_qc(out_dir)
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

    else:
        self_path_str = st.text_input("Self image path", value="raw_faces/sub-001/self.jpg")
        other_path_str = st.text_input("Other image path", value="raw_faces/sub-001/other.jpg")

        if st.button("Run", type="primary"):
            if not subject_id:
                st.error("Subject ID is required.")
            else:
                self_path = resolve_path(self_path_str)
                other_path = resolve_path(other_path_str)
                if not self_path.exists() or not other_path.exists():
                    st.error("Self/other paths must exist.")
                else:
                    with st.spinner("Processing..."):
                        try:
                            out_dir = run_subject(
                                subject_id,
                                self_path,
                                other_path,
                                out_base,
                                overrides,
                                pipeline_flags,
                            )
                            st.success(f"Done. Output: {out_dir}")
                            show_qc(out_dir)
                        except Exception as exc:
                            st.error(f"Failed: {exc}")

else:
    st.subheader("Batch folder")
    st.write("Expected structure: root/sub-001/self.jpg and root/sub-001/other.jpg")

    root_dir_str = st.text_input("Batch root folder", value="raw_faces")
    self_name = st.text_input("Self filename", value="self.jpg")
    other_name = st.text_input("Other filename", value="other.jpg")

    if st.button("Scan"):
        root_dir = resolve_path(root_dir_str)
        st.session_state["batch_scan"] = scan_batch(root_dir, self_name, other_name)

    batch_scan = st.session_state.get("batch_scan", [])
    if batch_scan:
        df = pd.DataFrame(
            [
                {
                    "subject": row["subject"],
                    "self": str(row["self_path"]),
                    "other": str(row["other_path"]),
                    "valid": row["valid"],
                }
                for row in batch_scan
            ]
        )
        st.dataframe(df, use_container_width=True)

    if st.button("Run batch", type="primary"):
        root_dir = resolve_path(root_dir_str)
        if not root_dir.exists():
            st.error("Batch root folder does not exist.")
        else:
            subjects = scan_batch(root_dir, self_name, other_name)
            if not subjects:
                st.error("No subject folders found.")
            else:
                results = []
                progress = st.progress(0.0)
                for idx, row in enumerate(subjects):
                    if not row["valid"]:
                        results.append(
                            {
                                "subject": row["subject"],
                                "status": "missing files",
                                "out_dir": "",
                            }
                        )
                        progress.progress((idx + 1) / len(subjects))
                        continue

                    try:
                        out_dir = run_subject(
                            row["subject"],
                            row["self_path"],
                            row["other_path"],
                            out_base,
                            overrides,
                            pipeline_flags,
                        )
                        results.append(
                            {
                                "subject": row["subject"],
                                "status": "ok",
                                "out_dir": str(out_dir),
                            }
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "subject": row["subject"],
                                "status": f"failed: {exc}",
                                "out_dir": "",
                            }
                        )
                    progress.progress((idx + 1) / len(subjects))

                st.dataframe(pd.DataFrame(results), use_container_width=True)
