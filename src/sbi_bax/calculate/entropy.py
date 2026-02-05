# Compare settings where the true density is known to estimates using samples.
import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Dict, Any


def histogram_entropy(
    samples: np.ndarray, bins: int = 20, weights: Optional[np.ndarray] = None
) -> float:
    """
    Estimate the entropy of a distribution from samples using histogram.
    Args:
        samples (np.ndarray): Samples from the distribution.
        bins (int): Number of bins for the histogram.
    Returns:
        float: Estimated entropy.
    """
    hist, bin_edges = np.histogram(samples, bins=bins, density=True, weights=weights)
    hist += 1e-10  # Avoid log(0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return -np.trapz(hist * np.log(hist), bin_centers)


def kde_entropy(
    samples: np.ndarray, bandwidth: float = 0.1, grid_size: int = 10_000
) -> float:
    """
    Estimate the entropy of a distribution from samples using kernel density estimation.
    Args:
        samples (np.ndarray): Samples from the distribution.
        bandwidth (float): Bandwidth for the kernel density estimation.
    Returns:
        float: Estimated entropy.
    """
    kde = gaussian_kde(samples, bw_method=bandwidth)
    x = np.linspace(np.min(samples), np.max(samples), grid_size)
    pdf = kde(x)
    pdf += 1e-10  # Avoid log(0)
    return -np.trapz(pdf * np.log(pdf), x)


def entropy_inf_fun(dens_x: np.ndarray) -> np.ndarray:
    """
    Influence function for entropy estimation.

    Args:
        dens_x (np.ndarray): Density values at data points X

    Returns:
        np.ndarray: Influence function values
    """
    return -np.log(dens_x)


def entropy_asymp_var(dens_x: np.ndarray) -> float:
    """
    Asymptotic variance for entropy estimation.

    Args:
        dens_x (np.ndarray): Density values at data points X

    Returns:
        float: Asymptotic variance
    """
    log_dens_x = np.log(dens_x)
    return np.mean(log_dens_x**2) - np.mean(log_dens_x) ** 2


def inf_fun_entropy(
    X: np.ndarray,
    bandwidth: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float]:
    """
    Estimate Shannon entropy -∫ p log(p) using kernel density estimation.

    This is a Python translation of the MATLAB code from:
    https://github.com/kirthevasank/if-estimators/blob/master/estimators/shannonEntropy.m

    Args:
        X (np.ndarray): Data samples
        bandwidth (float, optional): KDE bandwidth. If None, uses Scott's rule
        params (dict, optional): Additional estimation parameters

    Returns:
        Tuple[float, float, float]: (entropy_estimate, asymptotic_analysis, bandwidth_used)
    """
    # Parse parameters (simplified version of parseOneDistroParams)
    if params is None:
        params = {}

    # Handle 1D case
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Estimate density using KDE
    if bandwidth is None:
        kde = gaussian_kde(X.T)  # gaussian_kde expects (n_features, n_samples)
    else:
        kde = gaussian_kde(X.T, bw_method=bandwidth)

    # Evaluate density at data points
    dens_x = kde(X.T)

    # Add small epsilon to avoid log(0)
    dens_x = np.maximum(dens_x, 1e-10)

    # Compute entropy estimate using influence function
    inf_fun_vals = entropy_inf_fun(dens_x)
    entropy_estimate = np.mean(inf_fun_vals)

    # # Compute asymptotic variance
    # asymp_var = entropy_asymp_var(dens_x)

    return entropy_estimate


def entropy_cdf(samples: np.ndarray):
    """
    # NOTE: not implemented with weights but vectorizable.
    Implements equation (16) from here:
    jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf
    """
    # Sort each dataset along axis=1 to get the empirical quantile function
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)  # Ensure samples is 2D
    # Sort each dataset (each row)
    X_sorted = np.sort(samples, axis=1)
    n = X_sorted.shape[1]
    # Define the window size m
    m = int(np.floor(np.sqrt(n)))

    # Compute spacings between points that are k apart.
    # This yields an array of shape (m, n-k)
    spacings = X_sorted[:, m:] - X_sorted[:, :-m]

    # Avoid log(0)
    spacings = np.clip(spacings, 1e-10, None)

    # Compute the entropy estimate for each dataset:
    entropies = np.mean(np.log(spacings * n / m), axis=1)

    return entropies.item() if entropies.size == 1 else entropies
