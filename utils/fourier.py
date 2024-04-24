import numpy as np
import matplotlib.pyplot as plt


def calculate_dft2(input_img):
    ft = np.fft.fft2(input_img)
    return np.fft.fftshift(ft)


def calculate_idft2(input_img):
    ifft = np.fft.ifftshift(input_img)
    return np.fft.ifft2(ifft).real


def plot_ft_spectrum(ft):
    plt.set_cmap("gray")
    plt.imshow(np.log(abs(ft)))
    plt.axis("off")


def apply_low_pass_filter(ft, threshold):
    mask = np.zeros(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 1

    return ft * mask


def apply_high_pass_filter(ft, threshold):
    mask = np.ones(ft.shape, np.uint8)
    rows, cols = ft.shape[0], ft.shape[1]
    crows, ccols = rows // 2, cols // 2
    mask[crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = 0

    return ft * mask


def apply_band_pass_filter(ft, lower_threshold, upper_threshold):
    assert lower_threshold < upper_threshold

    w = (upper_threshold - lower_threshold) // 2

    ft = apply_high_pass_filter(ft, lower_threshold - w)
    ft = apply_low_pass_filter(ft, upper_threshold + w)

    return ft
