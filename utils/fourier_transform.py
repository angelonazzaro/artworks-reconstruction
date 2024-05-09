import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def calculate_dft(input_signal: np.ndarray) -> np.ndarray:
    """
    Calculates the 1D Discrete Fourier Transform (DFT) of a 1D signal.

    Args:
    - input_img (np.ndarray): Input signal as a 1D numpy array.

    Returns:
    - ft (np.ndarray): 1D DFT of the input signal, Fourier coefficients shifted to the center.

    This function computes the 2D DFT of the input image using numpy's fft.fft() function.
    The resulting DFT coefficients are shifted to center the low frequencies.
    """
    assert len(input_signal.shape) == 1

    ft = np.fft.fft(input_signal)
    return np.fft.fftshift(ft)


def calculate_idft(input_img: np.ndarray) -> np.ndarray:
    """
    Calculates the 1D Inverse Discrete Fourier Transform (IDFT) of a 1D signal.

    Args:
    - input_img (np.ndarray): Input signal as a 1D numpy array.

    Returns:
    - ifft (np.ndarray): 1D IDFT of the input signal, real part of the inverse transform.

    This function computes the 1D IDFT of the input image using numpy's fft.ifft() function.
    The input signal is expected to have been shifted to center the low frequencies.
    """
    assert len(input_img.shape) == 1

    ifft = np.fft.ifftshift(input_img)
    return np.fft.ifft(ifft).real


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
    assert len(input_img.shape) == 2

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
    assert len(input_img.shape) == 2

    ifft = np.fft.ifftshift(input_img)
    return np.fft.ifft2(ifft).real


def plot_ft_spectrum(ft: np.ndarray, xlabel: str = None, ylabel: str = None, title: str = None, show: bool = True) \
        -> None:
    """
    Plots the spectrum of the 1D Fourier Transform.

    Args:
    - ft (np.ndarray): 1D Fourier Transform coefficients.

    This function plots the magnitude spectrum of the 1D Fourier Transform.
    The Fourier coefficients are expected to be shifted to center the low frequencies.
    """
    assert len(ft.shape) == 1

    # multiplying by 20 and adding 1 (avoid log(0)) to the real part of the complex magnitudes makes it easier
    # to plot the spectrum
    plt.plot(20 * np.log(np.abs(ft) + 1))
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(label=title)

    if show:
        plt.show()


def plot_2d_ft_spectrum(ft: np.ndarray, xlabel: str = None, ylabel: str = None, title: str = None, show: bool = True) \
        -> None:
    """
    Plots the spectrum of the 2D Fourier Transform.

    Args:
    - ft (np.ndarray): 2D Fourier Transform coefficients.

    This function plots the magnitude spectrum of the 2D Fourier Transform.
    The Fourier coefficients are expected to be shifted to center the low frequencies.
    """
    assert len(ft.shape) == 2

    # multiplying by 20 and adding 1 (avoid log(0)) to the real part of the complex magnitudes makes it easier
    # to plot the spectrum
    plt.imshow(20 * np.log(np.abs(ft) + 1), cmap='gray')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(label=title)
    plt.axis("off")

    if show:
        plt.show()


def apply_2d_low_pass_filter(ft: np.ndarray, threshold: int) -> np.ndarray:
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
    assert len(ft.shape) == 2

    mask = np.zeros(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 1
    return ft * mask


def apply_2d_high_pass_filter(ft: np.ndarray, threshold: int) -> np.ndarray:
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
    assert len(ft.shape) == 2

    mask = np.ones(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 0
    return ft * mask


def apply_2d_band_pass_filter(ft: np.ndarray, lower_threshold: int, upper_threshold: int) -> np.ndarray:
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
    assert lower_threshold < upper_threshold and len(ft.shape) == 2

    w = (upper_threshold - lower_threshold) // 2

    mask = np.zeros(ft.shape, dtype=np.uint8)
    crows, ccols = ft.shape[0] // 2, ft.shape[1] // 2
    center = (crows, ccols)

    # Creating upper bandpass mask
    cv.circle(mask, center, upper_threshold + w, 1, -1)
    upper_masked_ft = ft * mask

    # Creating lower bandpass mask
    mask.fill(0)  # Resetting mask
    cv.circle(mask, center, lower_threshold - w, 1, -1)
    lower_masked_ft = ft * mask

    # Returning the bandpass filtered spectrum
    return upper_masked_ft - lower_masked_ft

