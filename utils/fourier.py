import numpy as np
import matplotlib.pyplot as plt


def calculate_dft2(input_img: np.ndarray) -> np.ndarray:
    """
    Calculates the 2D Discrete Fourier Transform (DFT) of an image.

    Args:
    - input_img (np.ndarray): Input image as a 2D numpy array.

    Returns:
    - ft (np.ndarray): 2D DFT of the input image, Fourier coefficients shifted to the center.

    This function computes the 2D DFT of the input image using numpy's fft.fft2() function.
    The resulting DFT coefficients are shifted to center the low frequencies.
    """
    ft = np.fft.fft2(input_img)
    return np.fft.fftshift(ft)


def calculate_idft2(input_img: np.ndarray) -> np.ndarray:
    """
    Calculates the 2D Inverse Discrete Fourier Transform (IDFT) of an image.

    Args:
    - input_img (np.ndarray): Input image as a 2D numpy array.

    Returns:
    - ifft (np.ndarray): 2D IDFT of the input image, real part of the inverse transform.

    This function computes the 2D IDFT of the input image using numpy's fft.ifft2() function.
    The input image is expected to have been shifted to center the low frequencies.
    """
    ifft = np.fft.ifftshift(input_img)
    return np.fft.ifft2(ifft).real


def plot_ft_spectrum(ft: np.ndarray):
    """
    Plots the spectrum of the 2D Fourier Transform.

    Args:
    - ft (np.ndarray): 2D Fourier Transform coefficients.

    This function plots the magnitude spectrum of the 2D Fourier Transform.
    The Fourier coefficients are expected to be shifted to center the low frequencies.
    """
    plt.set_cmap("gray")
    plt.imshow(np.log(abs(ft)))
    plt.axis("off")


def apply_low_pass_filter(ft: np.ndarray, threshold: int) -> np.ndarray:
    """
    Applies a low-pass filter to the Fourier Transform.

    Args:
    - ft (np.ndarray): 2D Fourier Transform coefficients.
    - threshold (int): Threshold value for the filter.

    Returns:
    - filtered_ft (np.ndarray): Filtered Fourier Transform coefficients.

    This function applies a low-pass filter to the 2D Fourier Transform coefficients.
    Frequencies beyond the threshold are attenuated, while frequencies within the threshold are preserved.
    """
    mask = np.zeros(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 1
    return ft * mask


def apply_high_pass_filter(ft: np.ndarray, threshold: int) -> np.ndarray:
    """
    Applies a high-pass filter to the Fourier Transform.

    Args:
    - ft (np.ndarray): 2D Fourier Transform coefficients.
    - threshold (int): Threshold value for the filter.

    Returns:
    - filtered_ft (np.ndarray): Filtered Fourier Transform coefficients.

    This function applies a high-pass filter to the 2D Fourier Transform coefficients.
    Frequencies within the threshold are attenuated, while frequencies beyond the threshold are preserved.
    """
    mask = np.ones(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 0
    return ft * mask


def apply_band_pass_filter(ft: np.ndarray, lower_threshold: int, upper_threshold: int) -> np.ndarray:
    """
    Applies a band-pass filter to the Fourier Transform.

    Args:
    - ft (np.ndarray): 2D Fourier Transform coefficients.
    - lower_threshold (int): Lower frequency threshold for the filter.
    - upper_threshold (int): Upper frequency threshold for the filter.

    Returns:
    - filtered_ft (np.ndarray): Filtered Fourier Transform coefficients.

    This function applies a band-pass filter to the 2D Fourier Transform coefficients.
    Frequencies outside the specified range are attenuated, while frequencies within the range are preserved.
    """
    assert lower_threshold < upper_threshold
    w = (upper_threshold - lower_threshold) // 2
    ft = apply_high_pass_filter(ft, lower_threshold - w)
    ft = apply_low_pass_filter(ft, upper_threshold + w)
    return ft
