import numpy as np
from scipy import ndimage


def count_bright_peaks(image: np.ndarray, threshold: float = 1) -> int:
    """
    Count the number of bright peaks in a diffraction image.

    Args:
        image (np.ndarray): The diffraction image to analyze.
        threshold (float): Intensity threshold to consider a pixel as part of a peak.

    Returns:
        int: Number of bright peaks detected in the image.
    """
    # Threshold the image to find bright spots
    bright_spots = image > threshold
    # All images will be single-channel, so check if this is a single image
    if bright_spots.ndim == 2:
        bright_spots = bright_spots[np.newaxis, ...]
    # Label connected components
    num_features = np.zeros(bright_spots.shape[0], dtype=int)
    for i in range(bright_spots.shape[0]):
        _, num_features[i] = ndimage.label(bright_spots[i])
    return num_features
