import os
import shutil
import json

import cv2 as cv
import numpy as np

from typing import Tuple, List, Optional
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from feature_extraction.feature_extraction import compute_image_gradient
from preprocessing.edge_extraction import extract_working_region, filter_working_region


def restore_data(in_dir: str, output_dir: str):
    """
        Move all files from the input directory to the output directory.

        This function iterates over all files in the specified input directory (`in_dir`)
        and moves each file to the specified output directory (`output_dir`). If a file
        with the same name already exists in the output directory, it will be overwritten.

        Args:
            in_dir (str): The path to the input directory containing the files to be moved.
            output_dir (str): The path to the output directory where the files will be moved to.
    """
    for filename in os.listdir(in_dir):
        shutil.move(os.path.join(in_dir, filename), os.path.join(output_dir, filename))


def f1(precision_score: float, recall_score: float) -> float:
    """
    Calculates F1 score given precision and recall.

    Args:
        precision_score (float): Precision score.
        recall_score (float): Recall score.

    Returns:
        float: F1 score.
    """
    return 2 * (precision_score * recall_score) / (precision_score + recall_score) if (
            precision_score + recall_score) else 0


def recall(reference_image_id: int, root_dir: str, cluster_dirs_exp: List[str], ext: str = ".png") -> float:
    """
    Calculates recall given a reference image ID, root directory, and an excluded cluster directory.

    Args:
        reference_image_id (int): The ID of the reference image.
        root_dir (str): Root directory containing subdirectories.
        cluster_dirs_exp (List[str]): Names of the excluded cluster directories.
        ext (str, optional): File extension to filter images (default is ".png").

    Returns:
        float: Recall score (true positives / (true positives + false negatives)).
    """
    tp = 0
    for cluster_dir in cluster_dirs_exp:
        for root, _, files in os.walk(os.path.join(root_dir, cluster_dir)):
            for filename in files:
                if not filename.endswith(ext):
                    continue
                if filename.split(".")[1] == str(reference_image_id):
                    tp += 1

    fn = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in cluster_dirs_exp]
        for filename in filenames:
            if not filename.endswith(ext):
                continue
            if filename.split(".")[1] == str(reference_image_id):
                fn += 1
    total = tp + fn

    return tp / total if total else 0


def compute_metrics(reference_image_id: int, root_dir: str, ext: str = ".png", output_file: str = None) -> dict:
    """
    Calculates precision scores for each cluster directory given a reference image ID and a root directory containing images.
    Also calculates the recall for the cluster directory with the highest precision.

    Args:
        reference_image_id (int): The ID of the reference image.
        root_dir (str): Path to the directory containing the clusters.
        ext (str, optional): File extension to filter images (default is ".png").
        output_file (str, optional): Path to the file where metrics will be saved (default is None).

    Returns:
        dict: A dictionary containing:
            - "max_item": A tuple with the directory having the highest precision score and the score itself.
            - "precision_scores": A dictionary with precision scores for each cluster directory.
            - "recall": The recall score for the cluster directory with the highest precision.
    """
    f1_scores = {}
    first_dir = True

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if first_dir:
            first_dir = False
            continue
        tp = 0
        for filename in filenames:
            if not filename.endswith(ext):
                continue
            if filename.split(".")[1] == str(reference_image_id):
                tp += 1
        precision_score = tp / len(filenames) if filenames else 0
        recall_score = recall(reference_image_id=reference_image_id, root_dir=root_dir,
                              cluster_dirs_exp=[dirpath.split(os.path.sep)[-1]], ext=ext)
        f1_scores[dirpath.split(os.path.sep)[-1]] = f1(precision_score, recall_score)

    max_value = max(f1_scores.values())
    cluster_dirs = [dirpath for dirpath, score in f1_scores.items() if score == max_value]
    max_items = [(dirpath, score) for dirpath, score in f1_scores.items() if score == max_value]

    metrics = {
        "max_items": max_items,
        "f1_scores": f1_scores,
    }

    if output_file is not None:
        json.dump(metrics, open(output_file, "w"))
    return metrics


def compute_color_histograms(images: list, image_ref: np.ndarray = None, flatten: bool = True) -> (
        List[np.ndarray] | Tuple[List[np.ndarray], Optional[np.ndarray]]):
    """
    Computes the color histograms for a list of images.

    Args:
        images (list): List of images to process.
        image_ref: Image reference.
        flatten (bool): Whether to flatten the histograms. Defaults to True.
    Returns:
        (list, np.ndarray): List of histograms for each image. Histogram color of image reference.
    """
    histograms_fragments = []
    for image in tqdm(images, desc="Computing color histograms"):
        # Calculate the color histogram for the image
        hist_src = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv.normalize(hist_src, hist_src, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        if flatten:
            hist_src = hist_src.flatten()

        histograms_fragments.append(hist_src)

    if image_ref is not None:
        hist_image_ref = cv.calcHist([image_ref], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv.normalize(hist_image_ref, hist_image_ref, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        return histograms_fragments, hist_image_ref

    return histograms_fragments


def compute_jacobians(images: list, image_ref: np.ndarray = None, flatten: bool = True) -> (
        List[np.ndarray] | Tuple[List[np.ndarray], Optional[np.ndarray]]):
    """
    Computes the Jacobians for a list of images.

    Args:
        images (list): List of images to process.
        image_ref (np.ndarray): Image Ref.
        flatten (bool): Whether to flatten the Jacobians. Defaults to True.

    Returns:
        list: List of Jacobians for each image.
    """

    def compute_jacobian(image: np.ndarray, max_width: int, max_height: int, flatten: bool) -> np.ndarray:
        # Resize the image to the maximum dimensions
        reshaped_image = cv.resize(image, (max_width, max_height))
        reshaped_image = cv.cvtColor(reshaped_image, cv.COLOR_BGR2HSV)

        # Compute the image gradient (Jacobians)
        jacobian = compute_image_gradient(reshaped_image)

        if flatten:
            jacobian = jacobian.flatten()

        return jacobian

    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    if image_ref is not None:
        max_width = max(max_width, image_ref.shape[1])
        max_height = max(max_height, image_ref.shape[0])

    jacobians_fragments = []
    for image in tqdm(images, desc="Computing Jacobians"):
        jacobian = compute_jacobian(image, max_width, max_height, flatten)
        jacobians_fragments.append(jacobian)

    jacobian_image_ref = None

    if image_ref is not None:
        jacobian_image_ref = compute_jacobian(image_ref, max_width, max_height, flatten)

    if image_ref is not None:
        return jacobians_fragments, jacobian_image_ref
    return jacobians_fragments


def reshape_jacobians(jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape the Jacobian arrays.

    Args:
        jacobian (np.ndarray): Array containing Jacobian information.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Reshaped arrays for gx_r, gx_g, gx_b, gx_gray, gy_r, gy_g, gy_b, gy_gray.
    """
    gx = jacobian[0][0].reshape(1, -1)
    gx_gray = jacobian[0][1].reshape(1, -1)
    gy = jacobian[1][1].reshape(1, -1)
    gy_gray = jacobian[1][1].reshape(1, -1)

    return gx, gx_gray, gy, gy_gray


def compute_color_histogram_dist_matrix(histograms: list, histogram_image_ref: np.ndarray = None) -> np.ndarray:
    """
       Computes the distance matrix based on color histograms of input images.

       Args:
           histogram_image_ref (np.ndarray): Image Ref color histogram
           histograms (np.ndarray): Array of color histograms.

       Returns:
           np.ndarray: Similarity matrix.
    """

    def compute_histograms_similarity(histogram1: np.ndarray, histogram2: np.ndarray) -> float:
        correlation = cv.compareHist(histogram1, histogram2, cv.HISTCMP_CORREL)
        # Chi-square distance is sensitive to differences in the shape of histograms.
        # It's useful when we want to capture differences in the color distribution between images.
        chi_square_distance = cv.compareHist(histogram1, histogram2, cv.HISTCMP_CHISQR)
        # Intersection measures the overlap between histograms.
        # It's useful when we want to capture how much two histograms share common values
        # (objects observed from different view points)
        intersection = cv.compareHist(histogram1, histogram2, cv.HISTCMP_INTERSECT) / np.sum(histogram1)

        # correlation and intersection are measures where higher values indicate higher distance while
        # chi-square distance, on the other hand, is a measure where a smaller value indicates higher distance.
        # A smaller chi-square value indicates less difference between the histograms. By negating it,
        # we penalize dissimilar histograms
        distance = 0.5 * correlation + 0.25 * intersection - 0.25 * chi_square_distance

        # technically it is a similarity measure, so we need to maintain it positive if we
        # intend to use it as a distance measure
        if distance < 0:
            distance = -distance

        return distance

    # calculate matrix distance
    distance_matrix = np.zeros((len(histograms), len(histograms)))  # Initialize distance matrix

    for i in (tqdm(range(len(histograms)), desc="Calculating similarities")):
        for j in range(i + 1, len(histograms)):
            # distance between histograms[i] and histograms[j]
            distance = compute_histograms_similarity(histograms[i], histograms[j])

            if histogram_image_ref is not None:
                distance_hist_i_ref = compute_histograms_similarity(histograms[i], histogram_image_ref)
                distance_hist_j_ref = compute_histograms_similarity(histograms[j], histogram_image_ref)
                distance = (distance * 0.25 + distance_hist_i_ref * 0.5 + distance_hist_j_ref * 0.5)

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def compute_jacobians_dist_matrix(jacobians: list, jacobian_image_ref: np.ndarray = None, combine="mean",
                                  metric: str = 'euclidean'):
    """
        Computes the distance matrix based on image gradients (Jacobians) of input images.

        Args:
            jacobians (np.ndarray): List of image gradient jacobians.
            jacobian_image_ref (np.ndarray): Image Reference gradient jacobian.
            combine (str): Method to combine distances ('mean' or 'median').
            metric (str): Metric to use for computing pairwise distances.

        Returns:
            np.ndarray: Similarity matrix.
    """

    def compute_jacobians_distance(gx: np.ndarray, gx_gray: np.ndarray, gy: np.ndarray, gy_gray: np.ndarray,
                                   gx_2: np.ndarray, gx_gray_2: np.ndarray, gy_2: np.ndarray,
                                   gy_gray_2: np.ndarray) -> float:
        dist_gx = pairwise_distances(gx, gx_2, metric=metric)
        dist_gy = pairwise_distances(gy, gy_2, metric=metric)
        dist_gx_gray = pairwise_distances(gx_gray, gx_gray_2, metric=metric)
        dist_gy_gray = pairwise_distances(gy_gray, gy_gray_2, metric=metric)

        distance = 0
        if combine == "mean":
            distance = np.mean([dist_gx, dist_gy, dist_gx_gray, dist_gy_gray])
        elif combine == "median":
            distance = np.median([dist_gx, dist_gy, dist_gx_gray, dist_gy_gray])

        return distance

    if combine not in ['mean', 'median']:
        raise ValueError("combine must be 'mean' or 'median'")

    distance_matrix = np.zeros((len(jacobians), len(jacobians)))

    for i in tqdm(range(len(jacobians)), desc="Calculating similarities"):
        gx, gx_gray, gy, gy_gray = reshape_jacobians(jacobians[i])
        for j in range(i + 1, len(jacobians)):
            gx_2, gx_gray_2, gy_2, gy_gray_2 = reshape_jacobians(jacobians[j])
            # distance between jacobians[i] and jacobians[j]
            distance = compute_jacobians_distance(gx, gx_gray, gy, gy_gray, gx_2, gx_gray_2, gy_2, gy_gray_2)

            if jacobian_image_ref is not None:
                gx_ref, gx_gray_ref, gy_ref, gy_gray_ref = reshape_jacobians(jacobian_image_ref)
                distance_jacobians_i_ref = compute_jacobians_distance(gx, gx_gray, gy, gy_gray, gx_ref, gx_gray_ref,
                                                                      gy_ref, gy_gray_ref)
                distance_jacobians_j_ref = compute_jacobians_distance(gx_2, gx_gray_2, gy_2, gy_gray_2, gx_ref,
                                                                      gx_gray_ref, gy_ref, gy_gray_ref)
                distance = (distance * 0.25 + distance_jacobians_i_ref * 0.5 + distance_jacobians_j_ref * 0.5)

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
    for filename in tqdm(os.listdir(img_dir), desc="Creating dataset"):
        if not filename.lower().endswith(img_ext):
            continue

        image = cv.imread(os.path.join(img_dir, filename), cv.IMREAD_UNCHANGED)
        if extract_borders:
            image = filter_working_region(extract_working_region(image, threshold=threshold))
        denoised_image = cv.fastNlMeansDenoisingColored(image)
        images.append(denoised_image)

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
