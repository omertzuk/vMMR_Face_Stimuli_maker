"""Tests for create_morph_sequence.py.

Run with:
    cd stimulus_generation
    python -m pytest tests/test_morph_sequence.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import create_morph_sequence as cms


class TestIntermediateAlphas:
    def test_zero_intermediates(self):
        assert cms.compute_intermediate_alphas(0) == []

    def test_single_intermediate_is_half(self):
        assert cms.compute_intermediate_alphas(1) == [0.5]

    def test_three_intermediates_are_quarters(self):
        assert cms.compute_intermediate_alphas(3) == [0.25, 0.5, 0.75]

    def test_step_is_deterministic_for_five_intermediates(self):
        alphas = cms.compute_intermediate_alphas(5)
        expected = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
        assert alphas == expected

    def test_negative_intermediates_raise(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            cms.compute_intermediate_alphas(-1)


class TestVideoSequence:
    def _frame(self, label: str, alpha: float) -> cms.SequenceFrame:
        return cms.SequenceFrame(
            label=label,
            alpha=alpha,
            path=Path(f"{label}.png"),
            image=np.zeros((4, 5), dtype=np.uint8),
        )

    def test_full_video_sequence_includes_endpoints_in_order(self):
        self_frame = self._frame("self", 0.0)
        morph_frames = [
            self._frame("morph_001", 0.25),
            self._frame("morph_002", 0.5),
        ]
        other_frame = self._frame("other", 1.0)

        sequence = cms.build_full_video_sequence(self_frame, morph_frames, other_frame)

        assert [frame.label for frame in sequence] == [
            "self",
            "morph_001",
            "morph_002",
            "other",
        ]
        assert [frame.alpha for frame in sequence] == [0.0, 0.25, 0.5, 1.0]

    def test_video_timing_uses_full_frame_count(self):
        seconds_per_frame, fps = cms.video_timing(num_frames=5, total_duration=2.0)
        assert seconds_per_frame == 0.4
        assert fps == 2.5

    def test_video_timing_validates_inputs(self):
        with pytest.raises(ValueError, match="num_frames"):
            cms.video_timing(num_frames=0, total_duration=2.0)
        with pytest.raises(ValueError, match="total_duration"):
            cms.video_timing(num_frames=3, total_duration=0)

    def test_write_sequence_video_writes_bgr_frames(self, tmp_path, monkeypatch):
        writers = []

        class FakeWriter:
            def __init__(self, path, fourcc, fps, frame_size, is_color):
                self.path = path
                self.fourcc = fourcc
                self.fps = fps
                self.frame_size = frame_size
                self.is_color = is_color
                self.frames = []
                self.released = False
                writers.append(self)

            def isOpened(self):
                return True

            def write(self, frame):
                self.frames.append(frame.copy())

            def release(self):
                self.released = True

        monkeypatch.setattr(cms.cv2, "VideoWriter", FakeWriter)
        monkeypatch.setattr(cms.cv2, "VideoWriter_fourcc", lambda *args: 1234)

        frames = [
            np.zeros((4, 5), dtype=np.uint8),
            np.full((4, 5), 128, dtype=np.uint8),
            np.full((4, 5), 255, dtype=np.uint8),
        ]
        info = cms.write_sequence_video(
            frames,
            tmp_path / "sequence.mp4",
            total_duration=1.5,
            overwrite=False,
        )

        assert info["frame_count"] == 3
        assert info["seconds_per_frame"] == 0.5
        assert info["fps"] == 2.0
        assert len(writers) == 1
        assert writers[0].frame_size == (5, 4)
        assert writers[0].fps == 2.0
        assert writers[0].released
        assert len(writers[0].frames) == 3
        assert all(frame.shape == (4, 5, 3) for frame in writers[0].frames)
