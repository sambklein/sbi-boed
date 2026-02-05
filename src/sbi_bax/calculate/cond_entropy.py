import torch
from typing import Callable


def entropy_at_x(
    x: torch.Tensor,
    outer_thetas: torch.Tensor,
    inner_thetas: torch.Tensor,
    simulator: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    density_function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    n_mc: int,
    noise_sim: float,
) -> float:
    """
    Calculate the entropy of the output given the input and parameters at a specific input x.
    This function computes H(y_x | D_t) or H(y_x | e_A) depending on how the thetas are sampled.
    It uses Monte Carlo sampling to estimate the entropy.

    Args:
        x (torch.Tensor): A scalar tensor representing a single instance of the design parameters.
        outer_thetas (torch.Tensor): Outer thetas sampled from the posterior.
        inner_thetas (torch.Tensor): Inner thetas sampled from the posterior.
        simulator (Callable): A function that simulates data given x, theta, and noise.
        density_function (Callable): A function that computes the density of y given x and theta.
        n_mc (int): Number of Monte Carlo samples.
        noise_sim (float): Noise level for the simulator.

    Returns:
        float: The estimated entropy.
    """
    # Ensure x is a 1D tensor
    if x.shape[0] != 1:
        raise ValueError("x must be a 1D tensor of design values.")
    # Get the number of features in x
    n_features = x.shape[1] if x.dim() > 1 else 1
    # Build a store for the log probabilities
    lp_yd = []
    for theta in outer_thetas:
        # Generate samples from the simulator
        samples = simulator(x.expand(n_mc, -1), theta, noise=noise_sim)
        # Get the number of inner thetas
        n_theta = len(inner_thetas)
        # Calculate the full density in one go
        p_yx = density_function(
            samples[:, :1]  # Extract the output values
            .repeat_interleave(n_theta)  # Repeat to match the number of inner thetas
            .view(-1, 1),  # Reshape to match the expected input
            samples[:, 1:]  # Repeat the above steps but for the input x
            .repeat_interleave(n_theta)
            .view(-1, n_features),
            inner_thetas.repeat(
                n_mc, 1
            ),  # Repeat the inner thetas to match the number of y_x
        )
        # Average over the thetas for every y_x and take the log
        lp_yd += [p_yx.view(n_mc, -1).mean(dim=1).log().mean()]
    return -torch.tensor(lp_yd).mean().item()
