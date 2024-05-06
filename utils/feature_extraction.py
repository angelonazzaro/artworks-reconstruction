import cv2 as cv
import numpy as np


def compute_gradient_per_channel(input_img: np.ndarray, combine: str = "max") -> (np.ndarray, np.ndarray):
    """
    Compute gradients for each channel of the input image and combine them based on the specified method.

    Args:
        input_img (np.ndarray): Input image as a NumPy array.
        combine (str, optional): Method to combine gradients for each channel.
                                  - "max": Take the maximum gradient among all channels.
                                  - "mean": Take the mean gradient among all channels.
                                  - "sum": Sum the gradients across all channels.
                                  - "median": Take the median gradient among all channels.
                                  Defaults to "max".

    Returns:
        tuple: A tuple containing:
               - gx (np.ndarray): Gradient along the x-axis.
               - gy (np.ndarray): Gradient along the y-axis.

    Raises:
        ValueError: If an invalid combine method is provided.
    """
    assert (len(input_img.shape) == 3)
    # b, g, r are arbitrary channel names
    b, g, r = cv.split(input_img)

    gx_r, gy_r = np.gradient(r)
    gx_g, gy_g = np.gradient(g)
    gx_b, gy_b = np.gradient(b)

    if combine == "max":
        gx = np.maximum.reduce([gx_r, gx_b, gx_g])
        gy = np.maximum.reduce([gy_r, gy_b, gy_g])
    elif combine == "mean":
        gx = np.mean.reduce([gx_r, gx_b, gx_g])
        gy = np.mean.reduce([gy_r, gy_b, gy_g])
    elif combine == "sum":
        gx = np.sum.reduce([gx_r, gx_b, gx_g])
        gy = np.sum.reduce([gy_r, gy_b, gy_g])
    elif combine == "median":
        gx = np.median.reduce([gx_r, gx_b, gx_g])
        gy = np.median.reduce([gy_r, gy_b, gy_g])
    else:
        raise ValueError("Invalid combine method. Choose between 'max', 'mean', 'sum', and 'median'.")

    return gx, gy


def generate_angular_hist(input_img: np.ndarray, n_bins: int = 9, _n: int = 20, combine: str = "max",
                          normalize: bool = True) -> (np.ndarray, np.ndarray):
    """
    Generate Angular Histogram features from an input image.

    Args:
        input_img (np.ndarray): Input image as a NumPy array.
        n_bins (int, optional): Number of bins in the histogram. Defaults to 9.
        _n (float, optional): Percentage of pixels to select among the total pixels with the largest magnitude values. Defaults to 20%.
        combine (str, optional): Method to combine gradients for each channel.
                                  - "max": Take the maximum gradient among all channels.
                                  - "mean": Take the mean gradient among all channels.
                                  - "sum": Sum the gradients across all channels.
                                  - "median": Take the median gradient among all channels.
        normalize (bool, optional): Whether to normalize the histogram. Defaults to True.

    Returns:
        tuple: A tuple containing:
               - hist (np.ndarray): The computed histogram of oriented gradients.
               - top_mags (np.ndarray): The magnitudes of the top 'n' pixels.
    """
    angle_step = 180 / n_bins
    bins = np.arange(0, 180, angle_step)

    if len(input_img.shape) > 2:
        gx, gy = compute_gradient_per_channel(input_img[:, :, :3], combine=combine)
    else:
        gx, gy = np.gradient(input_img)

    mag = np.hypot(gx, gy)  # Magnitudes
    angles = np.rad2deg(np.arctan2(gy, gx))  # Angles

    N = input_img.shape[0] * input_img.shape[1]
    n = N * _n // 100

    # Select top 'n' pixels with the largest magnitude values
    sorted_indices = np.argsort(mag)[::-1]  # Sort indices in descending order
    top_mags = mag.ravel()[sorted_indices[:n]]
    top_angles = angles.ravel()[sorted_indices[:n]]

    # Create histogram
    hist, _ = np.histogram(top_angles, bins=bins, range=(0, 180))
    # Inserts a value of 0 at index 0 of the hist array to ensure all bins are represented
    hist = np.insert(hist, 0, 0)[:n_bins]
    hist = hist.astype(float)

    # Normalize the histogram
    if normalize:
        hist /= np.sum(hist)

    return hist, top_mags
