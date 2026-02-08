import cv2
import numpy as np
from pathlib import Path


def preprocess(prev_frame: str, current_frame: str) -> list:
    """Utility function to preprocess high-dimensional frame into grayscale and downscaled frame

    Logic:
        - Read both frames into a 3D numpy array (rows x cols x channels)
        - Take the maximum pixel values (element-wise) between 2 frames
        - Convert the new frame into a grayscaled one
        - Resize the frame into a 84x84 2D numpy array

    Args:
        prev_frame (str): The path of the previous frame
        current_frame (str): The path of the current frame

    Returns:
        array: a 2D array (84 x 84)
    """
    prev_img = cv2.imread(str(prev_frame))
    current_img = cv2.imread(str(current_frame))

    new_frame = np.maximum(prev_img, current_img)

    gray_img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    rescaled_img = cv2.resize(gray_img, (84, 84), interpolation=cv2.INTER_AREA)
    return rescaled_img
