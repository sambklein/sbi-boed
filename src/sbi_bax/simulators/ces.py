import torch
import torch.nn as nn

from sbi_bax.utils.torch_utils import get_device
from sbi_bax.utils.distributions import CensoredSigmoidNormal


class Ces(nn.Module):
    def __init__(
        self,
        noise_scale: float = 0.005,  # sigma_eta in the model
        epsilon: float = 2 ** (-22),  # epsilon for response clipping
        device=None,
    ) -> None:
        super().__init__()
        self.device = device or get_device()

        self.basket_dim = 3  # Each basket has 3 commodities
        self.noise_scale = noise_scale
        self.epsilon = epsilon

    def utility(self, x, rho, alpha):
        """Calculate CES utility for a basket

        Args:
            x: basket of goods [B, basket_dim]
            rho: elasticity parameter [B, 1]
            alpha: weights for each good [B, basket_dim]

        Returns:
            utility value [B, 1]
        """
        x_pow_rho = x**rho

        # Compute weighted sum
        weighted_sum = torch.sum(alpha * x_pow_rho, dim=-1, keepdim=True)

        # Compute utility U(x) = (weighted_sum)^(1/rho)
        utility = weighted_sum ** (1.0 / rho)

        return utility

    def get_params(self, xi, theta):
        # Extract parameters
        rho = theta[..., 0:1]  # [B, 1]

        if theta.shape[-1] == 4:
            # Enforce alpha sums to 1
            alpha12 = theta[..., 1:3]  # [B, 2]
            alpha3 = 1.0 - torch.sum(alpha12, dim=-1, keepdim=True)  # [B, 1]
            alpha = torch.cat([alpha12, alpha3], dim=-1)  # [B, 3]
            log_u = theta[..., 3:4]  # [B, 1]
        elif theta.shape[-1] == 5:
            alpha = theta[..., 1:4]  # [B, 3]
            log_u = theta[..., 4:5]  # [B, 1]
        u = torch.exp(log_u)

        # Split the input into two baskets
        xi = torch.clamp(xi, min=0.01, max=100.0)
        basket1 = xi[..., : self.basket_dim]  # [B, 3]
        basket2 = xi[..., self.basket_dim :]  # [B, 3]

        # Calculate utility for each basket
        u1 = self.utility(basket1, rho, alpha)  # [B, 1]
        u2 = self.utility(basket2, rho, alpha)  # [B, 1]

        # Calculate utility difference
        utility_diff = u1 - u2  # [B, 1]

        # Calculate the mean and std of the response distribution
        mu_eta = utility_diff * u  # [B, 1]

        # Calculate the standard deviation (noise level)
        basket_diff = basket1 - basket2
        basket_dist = torch.norm(basket_diff, dim=-1, p=2, keepdim=True)  # [B, 1]
        sigma_eta = (1 + basket_dist) * self.noise_scale * u  # [B, 1]
        return mu_eta, sigma_eta

    def __call__(self, xi, theta, nuisance=None, noiseless=False):
        """Simulate CES preference ratings
        Args:
            xi: basket pairs [B, 6]
            theta: parameters [B, 5]
            nuisance: unused (for API compatibility)
            noiseless: if True, raise NotImplementedError
        Returns:
            y: preference rating [B, 1]
        """
        mu_eta, sigma_eta = self.get_params(xi, theta)
        if noiseless:
            raise NotImplementedError("Noiseless mode is not implemented for CES.")
        else:
            samples = CensoredSigmoidNormal(
                mu_eta, sigma_eta, self.epsilon, 1 - self.epsilon, device=self.device
            ).rsample()
            if torch.isnan(samples).any():
                raise ValueError("NaN encountered in CES simulation.")
            return samples

    def log_prob(self, y, xi, theta, collapse=True):
        """Calculate log likelihood of observation

        Args:
            y: preference rating [B, 1]
            xi: basket pairs [B, 6]
            theta: parameters [B, 5]

        Returns:
            log likelihood [B, 1]
        """
        mu_eta, sigma_eta = self.get_params(xi, theta)
        log_prob = CensoredSigmoidNormal(
            mu_eta, sigma_eta, self.epsilon, 1 - self.epsilon, device=mu_eta.device
        ).log_prob(y)
        if torch.isnan(log_prob).any():
            raise ValueError("NaN encountered in log_prob computation.")
        return log_prob


if __name__ == "__main__":
    from sbi_bax.data.ces import CesPrior

    # Example usage
    simulator = Ces()
    B = 4  # batch size

    # Single observation per theta-design pair
    xi = torch.rand(B, 6) * 100  # B designs
    print(xi)
    theta = CesPrior().sample((B,))  # B parameter sets

    y = simulator(xi, theta)  # [B, 1]
    log_p = simulator.log_prob(y, xi, theta)  # [B, 1]

    print(f"y shape: {y.shape}, {y}")
    print(f"log_prob shape: {log_p.shape}, {log_p}")
