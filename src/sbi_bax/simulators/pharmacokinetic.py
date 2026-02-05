import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from sbi_bax.data.pharmacokinetic import PharmacoPrior
from sbi_bax.utils.torch_utils import get_device


class Pharmacokinetic(nn.Module):
    """
    Pharmacokinetic simulator
    Implements the model:
        mean = (D_v / V) * (k_a / (k_a - k_e)) * (exp(-k_e * t) - exp(-k_a * t))
        y ~ Normal(mean, sd)
    """

    def __init__(
        self,
        D_v=400.0,
        epsilon_scale=math.sqrt(0.01),
        nu_scale=math.sqrt(0.1),
        device=None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.p = 3  # latent dimension: [k_a, k_e, V]

        # Experimental constants NOTE: these should be treated as nuisance params.
        self.D_v = D_v
        self.epsilon_scale = epsilon_scale
        self.nu_scale = nu_scale

    def __call__(self, xi, thetas, nuisance=None, noiseless=False):
        """
        Forward model:
            xi: observation times, shape [..., n_observations, 1]
            thetas: latent parameters [k_a, k_e, V], shape [..., 3]
        Returns: y measurements, shape [..., n_observations]
        """
        if thetas.dim() == 1:
            thetas = thetas.unsqueeze(0)
        if xi.dim() == 1:
            xi = xi.unsqueeze(-1)

        # Exponentiate to get actual parameters and unpack
        k_a, k_e, V = [thetas.exp()[..., [i]] for i in range(self.p)]
        assert (k_a > k_e).all(), "k_a must be greater than k_e"

        # Compute the mean concentration with explicit broadcasting
        scale = (self.D_v / V) * (k_a / (k_a - k_e))  # (..., 1)
        rate_e = (xi * k_e).squeeze(-1)  # (..., n_obs)
        rate_a = (xi * k_a).squeeze(-1)  # (..., n_obs)
        mean = scale.squeeze(-1) * (
            torch.exp(-rate_e) - torch.exp(-rate_a)
        )  # (..., n_obs)

        if noiseless:
            return mean
        else:
            # Compute heteroscedastic noise
            sd = torch.sqrt((mean * self.epsilon_scale) ** 2 + self.nu_scale**2)
            return dist.Normal(mean, sd).rsample()

    def log_prob(self, y_obs, xi, thetas, collapse=True):
        """
        Compute log-likelihood log p(y | xi, theta)
        """
        mean_y = self(xi, thetas, noiseless=True)
        sd = torch.sqrt((mean_y * self.epsilon_scale) ** 2 + self.nu_scale**2)
        return dist.Normal(mean_y.squeeze(), sd.squeeze()).log_prob(y_obs.squeeze())


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Fix all random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Instantiate simulator
    simulator = Pharmacokinetic()

    # Observation times (xi): uniform over 24 hours
    n_obs = 100
    xi = torch.linspace(0, 24, n_obs).unsqueeze(-1)

    # Sample a few latent thetas
    prior = PharmacoPrior()
    n_thetas = 3
    thetas = prior.sample((n_thetas,))

    # Simulate data
    y = simulator(
        xi.repeat(n_thetas, 1, 1).view(-1),
        thetas.repeat_interleave(n_obs, dim=0).view(-1, 3),
        noiseless=False,
    ).view(n_thetas, n_obs)

    print("Example theta:", thetas)
    print("Simulated y:", y)

    # Plot example concentration curve
    plt.figure()
    plt.plot(xi.squeeze(), y.squeeze().transpose(0, 1), "o", label="Simulated")
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title("Pharmacokinetic Simulation")
    plt.legend()
    plt.tight_layout()

    Path("figures/pharmacokinetic/").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/pharmacokinetic/pharmacokinetic_example.pdf", dpi=150)
    plt.close()
