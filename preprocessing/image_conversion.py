import cv2 as cv
import numpy as np


def convert_color_model(input_image: np.ndarray, color_model: int = cv.COLOR_BGR2HSV, keep_alpha: bool = True) \
        -> np.ndarray:
    """
    Convert an image from BGRA to any color model while preserving the alpha channel.

    Args:
    - input_image (np.ndarray): Input image in RGBA format (4 channels).
    - color_model (int): Color conversion code. Default is cv.COLOR_BGR2HSV.
    - keep_alpha (bool): Keep alpha channel. Default is True.

    Returns:
    - np.ndarray: Converted image with the specified color model and preserved alpha channel.
    """
    b, g, r, a = cv.split(input_image)
    bgr_image = cv.merge((b, g, r))
    color_model_image = cv.cvtColor(bgr_image, color_model)
    return cv.merge((color_model_image, a)) if keep_alpha else color_model_image


def convert_grayscale_to_rgba(input_image: np.ndarray, alpha_channel: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to RGBA format.

    Args:
        input_image (np.ndarray): Grayscale input image.
        alpha_channel (np.ndarray): Alpha channel for the RGBA image.

    Returns:
        np.ndarray: RGBA image with the input image replicated across RGB channels and the specified alpha channel.
    """
    return np.dstack((input_image, input_image, input_image, alpha_channel))

