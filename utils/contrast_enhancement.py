import numpy as np
import cv2 as cv


def enhance_contrast(input_img: np.ndarray, alpha_channel: np.ndarray,clip_limit=2.0, title_grid_size=(8, 8)) -> np.ndarray:
    """
    Enhances the contrast of an input image using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
        input_img (numpy.ndarray): Input image in BGR color format.
        alpha_channel (numpy.ndarray): Alpha channel information of an image in BGRA color format.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (tuple of int, optional): Size of grid for histogram equalization.
            Defaults to (8, 8).

    Returns:
        numpy.ndarray: Enhanced image in BGR color format.

    This function enhances the contrast of an input image using Contrast Limited Adaptive Histogram Equalization (CLAHE)
    technique.
    The function first converts the input image from BGR color space to LAB color space.
    It extracts the lightness channel (L-channel) from the LAB color space.
    CLAHE is applied to the L-channel to enhance the contrast.
    The CLAHE-enhanced L-channel is merged with the original chrominance channels (a and b).
    The enhanced image is converted back to the BGR color space and then to BGRA color space, in that way we won't lose
    the information contained in the alpha_channel.
    """
    lab = cv.cvtColor(input_img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
    cl = clahe.apply(l_channel)

    limg = cv.merge((cl, a, b))

    enhanced_image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    enhanced_image = cv.cvtColor(enhanced_image, cv.COLOR_BGR2BGRA)
    enhanced_image[:, :, 3] = alpha_channel

    return enhanced_image
