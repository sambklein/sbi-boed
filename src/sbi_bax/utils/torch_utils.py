import torch

import numpy as np

from contextlib import contextmanager
import logging

# Set up logging
log = logging.getLogger(__name__)


@contextmanager
def no_grad(model):
    """
    Context manager that temporarily disables gradients for all model parameters.
    When exited, restores their original requires_grad states.
    """
    # Save the current requires_grad state for all parameters
    requires_grad_backup = {p: p.requires_grad for p in model.parameters()}

    # Disable gradients
    for p in model.parameters():
        p.requires_grad = False

    try:
        yield model
    finally:
        # Restore original requires_grad states
        for p, req_grad in requires_grad_backup.items():
            p.requires_grad = req_grad


def convert_bounds(bounds):
    """Convert bounds to numpy array."""
    if isinstance(bounds, (list, tuple)):
        return np.array(bounds)
    elif isinstance(bounds, torch.Tensor):
        return bounds.cpu().numpy()
    else:
        raise ValueError("Bounds must be a list, tuple, or torch.Tensor.")


def lhs(n_samples, n_dim, lb=0, ub=1):
    # Step 1: create N intervals (as lower bounds)
    cut_lower = np.linspace(0, 1, n_samples, endpoint=False)

    # Step 2: sample within each interval
    u = np.random.rand(n_samples, n_dim)
    points = cut_lower[:, None] + u / n_samples  # shape: (n_samples, n_dim)

    # Step 3: independently permute each column
    for j in range(n_dim):
        points[:, j] = points[np.random.permutation(n_samples), j]

    # Check if lb and ub are arrays
    lb = convert_bounds(lb)
    ub = convert_bounds(ub)

    # Step 4: scale to [lb, ub]
    points = lb + points * (ub - lb)

    # Step 5: return a shuffled array of shape (n_samples, n_dim)
    return torch.Tensor(points)[np.random.permutation(n_samples)]


def get_device():
    """
    Get the current device (GPU or CPU) for PyTorch tensors.
    Returns:
        torch.device: The current device.
    """
    # pick the fastest available device on macOS → MPS, then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


def sample_or_load(file_path, from_scratch, sample_func, n_samples=5000):
    if file_path.exists() and not from_scratch:
        log.info(f"Loading samples from {file_path}")
        samples = torch.load(file_path)
        # Ensure we have the right number of samples
        if len(samples) >= n_samples:
            log.info(f"Using {n_samples} samples from {file_path}")
            indx = np.random.choice(len(samples), n_samples)
            samples = samples[indx]
        else:
            log.info(
                f"Only {len(samples)} samples found in {file_path}, need {n_samples}. Generating additional samples."
            )
            new_samples = sample_func(n_samples - len(samples))
            samples = torch.cat([samples, new_samples], dim=0)
            # Save the updated samples
            torch.save(samples, file_path)
    else:
        samples = sample_func(n_samples)
        # Save the samples
        torch.save(samples, file_path)
    return samples


def sample_theta(posterior, theta_file, from_scratch, n_samples=5000):
    def sample_func(n):
        with torch.no_grad():
            return posterior.sample((n,)).cpu().squeeze(0)

    return sample_or_load(theta_file, from_scratch, sample_func, n_samples)


def sample_nuisance(
    nuisance_estimator,
    thetas,
    data_obs,
    nuisance_file,
    from_scratch,
    n_samples=5000,
    is_first=False,
):
    def sample_func(n):
        with torch.no_grad():
            if is_first:
                # In round 1 this is just the prior
                return nuisance_estimator.sample((n,)).cpu()
            else:
                return nuisance_estimator.sample(
                    thetas,
                    data_obs.unsqueeze(0).repeat(len(thetas), 1, 1),
                    num_samples=1,
                ).cpu()

    return sample_or_load(nuisance_file, from_scratch, sample_func, n_samples)
