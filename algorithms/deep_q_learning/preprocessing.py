"""Frame preprocessing for DQN-style pipelines (Mnih et al., 2015).

Used when two RGB frame paths are available: element-wise max, grayscale, resize to 84×84.
For live environment observations, the notebook also defines single-frame preprocessing
(`preprocess_frame` on RGB arrays); that path is documented in ALGORITHM.md.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(prev_frame: str, current_frame: str) -> np.ndarray:
    """Reduce high-dimensional frames to a grayscale 84×84 array.

    Steps:
        - Read both frames as BGR arrays (H×W×3)
        - Take element-wise maximum (reduces flicker)
        - Convert to grayscale
        - Resize to 84×84 with area interpolation

    Args:
        prev_frame: Path to the previous frame image.
        current_frame: Path to the current frame image.

    Returns:
        uint8 array of shape (84, 84).
    """
    prev_img = cv2.imread(str(prev_frame))
    current_img = cv2.imread(str(current_frame))

    new_frame = np.maximum(prev_img, current_img)

    gray_img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    rescaled_img = cv2.resize(gray_img, (84, 84), interpolation=cv2.INTER_AREA)
    return rescaled_img
