import os
import shutil

import cv2 as cv
import numpy as np

from typing import Tuple
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from .feature_extraction import compute_image_gradient
from .edge_extraction import extract_working_region, filter_working_region


def reshape_jacobians(jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, \
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape the Jacobian arrays.

    Args:
        jacobian (np.ndarray): Array containing Jacobian information.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Reshaped arrays for gx_r, gx_g, gx_b, gx_gray, gy_r, gy_g, gy_b, gy_gray.
    """
    gx_r = jacobian[0][0].reshape(1, -1)
    gx_g = jacobian[0][1].reshape(1, -1)
    gx_b = jacobian[0][2].reshape(1, -1)
    gx_gray = jacobian[0][3].reshape(1, -1)
    gy_r = jacobian[1][0].reshape(1, -1)
    gy_g = jacobian[1][1].reshape(1, -1)
    gy_b = jacobian[1][2].reshape(1, -1)
    gy_gray = jacobian[1][3].reshape(1, -1)

    return gx_r, gx_g, gx_b, gx_gray, gy_r, gy_g, gy_b, gy_gray


def compute_color_histogram_dist_matrix(images: np.ndarray):
    """
       Computes the distance matrix based on color histograms of input images.

       Args:
           images (np.ndarray): Array of input images.

       Returns:
           np.ndarray: Similarity matrix.
    """
    histograms = []
    for image in images:
        # working_region = cv2.cvtColor(working_region, cv.COLOR_BGR2HSV)
        hist_src = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
        hist_dst = cv.normalize(hist_src, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        histograms.append(hist_dst)

    # calculate matrix distance
    distance_matrix = np.zeros((len(histograms), len(histograms)))  # Initialize distance matrix

    for i in (tqdm(range(len(histograms)), desc="Calculating similarities")):
        for j in range(i + 1, len(histograms)):
            correlation = cv.compareHist(histograms[i], histograms[j], cv.HISTCMP_CORREL)
            # Chi-square distance is sensitive to differences in the shape of histograms.
            # It's useful when we want to capture differences in the color distribution between images.
            chi_square_distance = cv.compareHist(histograms[i], histograms[j], cv.HISTCMP_CHISQR)
            # Intersection measures the overlap between histograms.
            # It's useful when we want to capture how much two histograms share common values
            # (objects observed from different view points)
            intersection = cv.compareHist(histograms[i], histograms[j], cv.HISTCMP_INTERSECT) / np.sum(histograms[i])

            # correlation and intersection are measures where higher values indicate higher distance while
            # chi-square distance, on the other hand, is a measure where a smaller value indicates higher distance.
            # A smaller chi-square value indicates less difference between the histograms. By negating it,
            # we penalize dissimilar histograms
            distance = 0.5 * correlation + 0.25 * intersection - 0.25 * chi_square_distance

            # technically it is a similarity measure, so we need to maintain it positive if we
            # intend to use it as a distance measure
            if distance < 0:
                distance = -distance

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def compute_jacobians_dist_matrix(images: np.ndarray, combine="mean", metric: str = 'euclidean'):
    """
        Computes the distance matrix based on image gradients (Jacobians) of input images.

        Args:
            images (np.ndarray): Array of input images.
            combine (str): Method to combine distances ('mean' or 'median').
            metric (str): Metric to use for computing pairwise distances.

        Returns:
            np.ndarray: Similarity matrix.
    """

    if combine not in ['mean', 'median']:
        raise ValueError("combine must be 'mean' or 'median'")

    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    jacobians = []
    for working_region in images:
        reshaped_working_region = cv.resize(working_region, (max_width, max_height))
        reshaped_working_region = cv.cvtColor(reshaped_working_region, cv.COLOR_BGR2HSV)
        jacobian = compute_image_gradient(reshaped_working_region)
        jacobians.append(jacobian)

    distance_matrix = np.zeros((len(images), len(images)))

    for i in tqdm(range(len(jacobians)), desc="Calculating similarities"):
        gx_r, gx_g, gx_b, gx_gray, gy_r, gy_g, gy_b, gy_gray = reshape_jacobians(jacobians[i])
        for j in range(i + 1, len(jacobians)):
            gx_r_2, gx_g_2, gx_b_2, gx_gray_2, gy_r_2, gy_g_2, gy_b_2, gy_gray_2 = reshape_jacobians(jacobians[j])

            dist_gx_r = pairwise_distances(gx_r, gx_r_2, metric=metric)
            dist_gx_g = pairwise_distances(gx_g, gx_g_2, metric=metric)
            dist_gx_b = pairwise_distances(gx_b, gx_b_2, metric=metric)
            dist_gx_gray = pairwise_distances(gx_gray, gx_gray_2, metric=metric)
            dist_gy_r = pairwise_distances(gy_r, gy_r_2, metric=metric)
            dist_gy_g = pairwise_distances(gy_g, gy_g_2, metric=metric)
            dist_gy_b = pairwise_distances(gy_b, gy_b_2, metric=metric)
            dist_gy_gray = pairwise_distances(gy_gray, gy_gray_2, metric=metric)

            if combine == "mean":
                distance = np.mean(
                    [dist_gx_r, dist_gx_g, dist_gx_b, dist_gx_gray, dist_gy_b, dist_gy_gray, dist_gy_r, dist_gy_g,
                     dist_gy_b])
            elif combine == "median":
                distance = np.median(
                    [dist_gx_r, dist_gx_g, dist_gx_b, dist_gx_gray, dist_gy_b, dist_gy_gray, dist_gy_r, dist_gy_g,
                     dist_gy_b])

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def create_dataset(img_dir: str, img_ext: str = "png", extract_borders: bool = True, threshold: int = 0) -> list:
    """
    Create a dataset from images in a directory.

    Args:
        img_dir (str): Path to the directory containing images.
        img_ext (str): Extension of the image files.
        extract_borders (bool): Whether to extract borders from images.
        threshold (int): Threshold for border extraction.

    Returns:
        list: List of images.
    """
    if not os.path.exists(img_dir):
        raise ValueError(f'Image directory {img_dir} does not exist!')

    images = []
    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(img_ext):
            continue

        image = cv.imread(os.path.join(img_dir, filename), cv.IMREAD_UNCHANGED)
        if extract_borders:
            image = filter_working_region(extract_working_region(image, threshold=threshold))
        images.append(image)

    return images


def create_cluster_dirs(data_dir: str, output_dir: str, labels: list, img_ext: str = "png"):
    """
    Create directories for each cluster and move images to the corresponding cluster directories.

    Args:
        data_dir: Directory where images are located.
        output_dir (str): Path to the output directory.
        labels (list): List of cluster labels corresponding to each image.
        img_ext (str): Image extension.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Remove existing output directory if it's not empty
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    filename_images = []
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(img_ext):
            continue
        filename_images.append(filename)

    # Create cluster directories and move images
    for idx, label in enumerate(tqdm(labels, desc='Creating cluster dirs')):
        # Determine the directory for the current cluster
        cluster_dir = os.path.join(output_dir, "unclustered" if label == -1 else f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Get the filename of the image
        filename_image = filename_images[idx]

        # Move the image to the corresponding cluster directory
        shutil.copy(os.path.join(data_dir, filename_image), os.path.join(cluster_dir, filename_image))
