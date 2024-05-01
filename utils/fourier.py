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


def alpha_edge_detector(input_img):
    rows, cols, _ = input_img.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)

    # Create an array to store the offsets for neighboring pixels
    neighbor_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for x in range(rows):
        for y in range(cols):
            # Skip if alpha is zero
            if input_img[x][y][3] == 0:
                continue

            # Iterate over neighboring pixels
            for offset_x, offset_y in neighbor_offsets:
                neighbor_x = x + offset_x
                neighbor_y = y + offset_y

                # Check if neighbor is out of bounds or has alpha zero
                if (not (0 <= neighbor_x < rows and 0 <= neighbor_y < cols)
                        or input_img[neighbor_x][neighbor_y][3] == 0):
                    edges[x][y] = 1
                    break
    return edges


def extract_edge_interpolation_region(input_img, threshold=5):
    rows, cols, channels = input_img.shape
    interpolation_region = np.zeros((rows, cols, channels), dtype=np.uint8)

    neighbor_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for x in range(rows):
        for y in range(cols):
            for offset_x, offset_y in neighbor_offsets:
                neighbor_x = x + offset_x
                neighbor_y = y + offset_y

                # Check if neighbor is out of bounds or has alpha zero and move in the opposite direction
                if (not (0 <= neighbor_x < rows and 0 <= neighbor_y < cols)
                        or input_img[neighbor_x][neighbor_y][3] == 0):

                    # Add all the valid pixel in [1, threshold] to the interpolation region
                    for t in range(1, threshold + 1):
                        if offset_x > 0 or offset_y > 0:
                            opposite_x = neighbor_x - t if offset_x != 0 else neighbor_x
                            opposite_y = neighbor_y - t if offset_y != 0 else neighbor_y
                        else:
                            opposite_x = neighbor_x + t if offset_x != 0 else neighbor_x
                            opposite_y = neighbor_y + t if offset_y != 0 else neighbor_y

                        # check if pixel is out of bounds
                        if not (0 <= opposite_x < rows and 0 <= opposite_y < cols):
                            break

                        # if inner pixel has content, add it to the interpolation region
                        if input_img[opposite_x][opposite_y][3] != 0:
                            interpolation_region[opposite_x][opposite_y] = input_img[opposite_x][opposite_y]

    return interpolation_region
