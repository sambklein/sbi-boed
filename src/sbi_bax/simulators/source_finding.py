# SB+ BAX for source finding example
import torch
import torch.distributions as dist
from torch import nn


class SourceFinding(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        noise_scale=0.5,  # this is the scale of the noise term
        physical_dim=2,  # physical dimension
        n_source=2,  # number of sources
        device=None,
    ):
        super().__init__()
        self.base_signal = base_signal
        self.max_signal = max_signal
        self.physical_dim = physical_dim
        self.n_source = n_source
        # Observations noise scale:
        self.noise_scale = (
            torch.Tensor([noise_scale])
            if noise_scale is not None
            else torch.tensor(1.0)
        )

    def log_prob(
        self,
        y_obs: torch.Tensor,
        xi: torch.Tensor,
        thetas: torch.Tensor,
        collapse: bool = True,
    ):
        """Get the log likelihood of observations"""
        # Get the number of measurements
        n_measured = xi.shape[-2]
        bs = xi.shape[1]
        n_contrast = thetas.shape[0]
        if y_obs.shape[0] != n_contrast:
            # Reshape designs and thetas to match
            xi = xi.repeat_interleave(n_contrast, 0).view(-1, self.physical_dim)
            thetas = thetas.repeat(1, n_measured, 1).view(
                -1, self.n_source, self.physical_dim
            )
            # Call the forward method without noise
            mean_y = self(xi, thetas, noiseless=True)
            # Get the likelihood under the output gaussian with this mean
            log_prob = dist.Normal(mean_y, self.noise_scale.to(mean_y.device)).log_prob(
                y_obs.squeeze()
            )
            return log_prob.view(n_contrast, bs, n_measured).sum(-1)
        else:
            # Call the forward method without noise
            mean_y = self(xi, thetas, noiseless=True)
            # Get the likelihood under the output gaussian with this mean
            log_prob = dist.Normal(mean_y, self.noise_scale.to(mean_y.device)).log_prob(
                y_obs.squeeze()
            )
            return log_prob.sum(-1) if collapse else log_prob

    def __call__(self, xi, thetas, nuisance=None, noiseless=False):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.

        Args:
            xi: detector locations, shape [n_detectors, physical_dim]
            thetas: source locations, shape [n_sources, physical_dim]
            nuisance: optional nuisance parameters

        Returns:
            measurements: shape [n_detectors]
        """
        # Expand dimensions for broadcasting
        # xi: [n_detectors, 1, physical_dim]
        # thetas: [1, n_sources, physical_dim]
        xi_expanded = xi.unsqueeze(1)  # [n_detectors, 1, physical_dim]
        if thetas.shape[-1] == self.n_source * self.physical_dim:
            # Reshape the thetas to have shape [..., n_sources, physical_dim]
            thetas = thetas.view(thetas.shape[0], self.n_source, self.physical_dim)
        if thetas.dim() == 2:
            thetas = thetas.unsqueeze(0)  # [1, n_sources, physical_dim]
        elif thetas.shape[0] != xi.shape[0] and thetas.shape[0] != 1:
            raise ValueError(
                "The first dimension of thetas must be 1 or match the number of detectors."
            )

        # Compute squared distances between all detector-source pairs
        # Result shape: [n_detectors, n_sources]
        sq_two_norm = (xi_expanded - thetas).pow(2).sum(axis=-1)

        # Add small number and take inverse (signal strength from each source)
        # Shape: [n_detectors, n_sources]
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)

        # Sum over all sources for each detector, add base signal and take log
        # Shape: [n_detectors]
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1))

        # Sample from Normal distribution
        # Shape: [n_detectors]
        if noiseless:
            return mean_y
        else:
            return dist.Normal(mean_y, self.noise_scale.to(mean_y.device)).rsample()


if __name__ == "__main__":
    # Example usage:
    physical_dim = 2
    # Leave all params as default, this matches set up from iDAD
    simulator = SourceFinding(physical_dim=physical_dim)
    # Simulate some data for detectors placed randomly in [0,1]^2
    n_detectors = 5
    xi = torch.rand(n_detectors, physical_dim)
    # Sample true source locations from prior
    true_thetas = simulator.theta_prior.sample()
    print("True source locations:\n", true_thetas)
    # Simulate measurements
    y = simulator(xi, true_thetas)
    print("Simulated measurements:\n", y)
    # Plot the setup
    import matplotlib.pyplot as plt
    from pathlib import Path

    dummy_dir = Path("figures/source_finding/")
    dummy_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        xi[:, 0], xi[:, 1], c=y, s=y * 100, cmap="viridis", alpha=0.7, label="Detectors"
    )
    plt.colorbar(label="Measurement Value")
    plt.scatter(
        true_thetas[:, 0],
        true_thetas[:, 1],
        c="red",
        marker="x",
        s=100,
        label="True Sources",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Source Finding Setup")
    plt.legend(frameon=False)
    plt.grid()
    plt.savefig(dummy_dir / "source_finding_setup.pdf", dpi=150, bbox_inches="tight")
    plt.close()
