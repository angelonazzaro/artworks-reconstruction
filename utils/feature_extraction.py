import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def compute_image_gradient(input_img: np.ndarray, combine: str = "concat") -> (np.ndarray, np.ndarray):
    """
    Compute gradients for each channel of the input image and combine them based on the specified method.

    Args:
        input_img (np.ndarray): Input image as a NumPy array.
        combine (str, optional): Method to combine gradients for each channel.
                                  - "concat": Return Jacobian of gradient for each channel.
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
    assert (len(input_img.shape) == 3) and input_img.shape[2] >= 3
    # b, g, r are arbitrary channel names
    b, g, r = cv.split(input_img if input_img.shape[2] == 3 else input_img[:, :, :3])

    gx_r, gy_r = np.gradient(r)
    gx_g, gy_g = np.gradient(g)
    gx_b, gy_b = np.gradient(b)

    # by considering both gradient of the grayscale image and the gradients of the RGB channels,
    # we combine structural information with color information
    grayscale_image = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    gx_gray, gy_gray = np.gradient(grayscale_image)

    if combine == "concat":
        return np.array([[gx_r, gx_b, gx_g, gx_gray], [gy_r, gy_b, gy_g, gy_gray]])
    elif combine == "max":
        gx = np.maximum.reduce([gx_gray, gx_r, gx_b, gx_g])
        gy = np.maximum.reduce([gy_gray, gy_r, gy_b, gy_g])
    elif combine == "mean":
        gx = np.mean([gx_gray, gx_r, gx_b, gx_g], axis=0)
        gy = np.mean([gy_gray, gy_r, gy_b, gy_g], axis=0)
    elif combine == "sum":
        gx = np.sum([gx_gray, gx_r, gx_b, gx_g], axis=0)
        gy = np.sum([gy_gray, gy_r, gy_b, gy_g], axis=0)
    elif combine == "median":
        gx = np.median([gx_gray, gx_r, gx_b, gx_g], axis=0)
        gy = np.median([gy_gray, gy_r, gy_b, gy_g], axis=0)
    else:
        raise ValueError("Invalid combine method. Choose between 'concat', 'max', 'mean', 'sum', and 'median'.")

    return gx, gy


def generate_angular_hist(input_img: np.ndarray, n_bins: int = 9, _n: int = 20, combine: str = "concat",
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
        gx, gy = compute_image_gradient(input_img[:, :, :3], combine=combine)
    else:
        gx, gy = np.gradient(input_img)

    mag = np.hypot(gx, gy)  # Magnitudes
    angles = np.rad2deg(np.arctan2(gy, gx))  # Angles

    N = input_img.shape[0] * input_img.shape[1]
    n = N * _n // 100

    # Select top 'n' pixels with the largest magnitude values
    # Sort indices in descending order
    sorted_indices = np.argsort(np.sum(mag, axis=1))[::-1]
    top_mags = mag[sorted_indices[:n]]
    top_angles = angles[sorted_indices[:n]]

    # Create histogram
    hist, _ = np.histogram(top_angles, bins=bins, range=(0, 180))
    # Inserts a value of 0 at index 0 of the hist array to ensure all bins are represented
    hist = np.insert(hist, 0, 0)[:n_bins]
    hist = hist.astype(float)

    # Normalize the histogram
    if normalize:
        hist /= np.sum(hist)

    return hist, top_mags


def plot_channel_histograms(image: np.ndarray, mask: np.ndarray, title: str = "Histogram"):
    """
    Plot histograms for each channel of the input image.

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Mask to apply on the image.
        title (str): Title for the plot (default is "Histogram").

    Returns:
        None
    """
    # Split the image into its respective channels
    chans = cv.split(image)

    # Define colors for plotting
    colors = ("b", "g", "r")

    # Create a new figure for plotting
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.ylim([0, 256])

    # Loop over the image channels
    for (chan, color) in zip(chans, colors):
        # Create a histogram for the current channel
        hist = cv.calcHist([chan], [0], mask, [256], [0, 256])

        # Plot the histogram
        plt.plot(hist, color=color)

    # Set the x-axis limit
    plt.xlim([0, 256])
