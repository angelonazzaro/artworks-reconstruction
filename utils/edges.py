import numpy as np


def alpha_edge_detector(input_img: np.ndarray) -> np.ndarray:
    """
    Detects edge pixels based on the alpha channel of an image.

    Args:
    - input_img (np.ndarray): Input image as a 3D numpy array (height x width x channels),
                              where channels include RGB and alpha.

    Returns:
    - edges (np.ndarray): Binary mask indicating edge pixels (1 for edge pixels, 0 otherwise),
                          same shape as input_img.

    Raises:
    - ValueError: If input_img does not have an alpha channel or is not a 3D array.

    The function iterates through each pixel in the input image and identifies edge pixels
    based on the alpha channel. A pixel is considered an edge pixel if its alpha value is non-zero
    and at least one of its neighboring pixels has an alpha value of zero.
    """

    if len(input_img.shape) != 3 or input_img.shape[2] < 4:
        raise ValueError("Input image must have an alpha channel.")

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


def extract_interpolation_region(input_img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Extracts the interpolation region around edge pixels in an image.

    Args:
    - input_img (np.ndarray): Input image as a 3D numpy array (height x width x channels),
                              where channels include RGB and alpha.
    - threshold (int): Threshold distance from edge pixels. Defaults to 5.

    Returns:
    - interpolation_region (np.ndarray): Interpolation region around edge pixels,
                                         same shape as input_img.

    Raises:
    - ValueError: If input_img does not have an alpha channel or is not a 3D array.

    The function iterates through each pixel in the input image and identifies edge pixels
    based on the alpha channel. It then extracts pixels within a specified distance from the edge
    (defined by the threshold) to form the interpolation region.
    """

    if len(input_img.shape) != 3 or input_img.shape[2] < 4:
        raise ValueError("Input image must have an alpha channel.")

    rows, cols, _ = input_img.shape
    interpolation_region = np.zeros_like(input_img)

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

                        # Check if pixel is out of bounds
                        if not (0 <= opposite_x < rows and 0 <= opposite_y < cols):
                            break

                        # If inner pixel has content, add it to the interpolation region
                        if input_img[opposite_x][opposite_y][3] != 0:
                            interpolation_region[opposite_x][opposite_y] = input_img[opposite_x][opposite_y]

    return interpolation_region

