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

    assert len(input_img.shape) == 3 and input_img.shape[2] >= 4

    edges = np.zeros_like(input_img)

    # Define the neighbor offsets for 4-connectivity
    neighbor_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for offset_x, offset_y in neighbor_offsets:
        # Shift the image array to obtain the neighbor pixels
        neighbor_x = np.roll(input_img, offset_x, axis=0)
        neighbor_y = np.roll(input_img, offset_y, axis=1)

        # Create a mask to identify pixels with alpha zero in the neighbors
        mask = (neighbor_x[:, :, 3] == 0) | (neighbor_y[:, :, 3] == 0)

        # Update the interpolation region with pixels satisfying the mask
        edges[mask] = input_img[mask]

    return edges


def extract_working_region(input_img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Extracts the working region around edge pixels in an image.

    Args:
    - input_img (np.ndarray): Input image as a 3D numpy array (height x width x channels),
                              where channels include RGB and alpha.
    - threshold (int): Threshold distance from edge pixels. Defaults to 5.

    Returns:
    - working_region (np.ndarray): Interpolation region around edge pixels,
                                         same shape as input_img.

    Raises:
    - ValueError: If input_img does not have an alpha channel or is not a 3D array.

    The function iterates through each pixel in the input image and identifies edge pixels
    based on the alpha channel. It then extracts pixels within a specified distance from the edge
    (defined by the threshold) to form the working region.
    """

    assert len(input_img.shape) == 3 and input_img.shape[2] >= 4

    working_region = np.zeros_like(input_img)

    # Define the neighbor offsets for 4-connectivity
    neighbor_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for offset_x, offset_y in neighbor_offsets:
        # Shift the image array to obtain the neighbor pixels
        neighbor_x = np.roll(input_img, offset_x, axis=0)
        neighbor_y = np.roll(input_img, offset_y, axis=1)

        # Create a mask to identify pixels with alpha zero in the neighbors
        edge_mask = (neighbor_x[:, :, 3] == 0) | (neighbor_y[:, :, 3] == 0)
        working_mask = edge_mask.copy()

        # Expand the mask to include pixels within the threshold distance
        for t in range(1, threshold + 1):
            opposite_x = np.roll(edge_mask, -t * offset_x, axis=0)
            opposite_y = np.roll(edge_mask, -t * offset_y, axis=1)

            working_mask = working_mask | (opposite_x | opposite_y)

        working_region[working_mask] = input_img[working_mask]

    return working_region

