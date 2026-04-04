"""Tests for frame preprocessing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from deep_q_learning.preprocessing import preprocess


def _write_bgr_png(path: Path, bgr: np.ndarray) -> None:
    assert cv2.imwrite(str(path), bgr)


@pytest.fixture
def tmp_two_frames(tmp_path: Path) -> tuple[Path, Path]:
    """Two 16×16 BGR images (different and identical cases use same helper)."""
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    a[:, :] = (10, 20, 30)
    b = np.zeros((16, 16, 3), dtype=np.uint8)
    b[:, :] = (100, 50, 200)
    pa = tmp_path / "a.png"
    pb = tmp_path / "b.png"
    _write_bgr_png(pa, a)
    _write_bgr_png(pb, b)
    return pa, pb


def test_preprocess_output_shape_and_dtype(tmp_two_frames: tuple[Path, Path]) -> None:
    pa, pb = tmp_two_frames
    out = preprocess(str(pa), str(pb))
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8


def test_preprocess_identical_frames_max_is_stable(tmp_path: Path) -> None:
    """Edge case: same path twice — max is identity; output still valid."""
    img = np.full((32, 32, 3), 42, dtype=np.uint8)
    p = tmp_path / "same.png"
    _write_bgr_png(p, img)
    out = preprocess(str(p), str(p))
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8
    assert np.all(out == 42)
