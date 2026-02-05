# SB+ BAX for crystal diffraction images
import torch


class NthOrderPolynomial:
    def __init__(self, noise_sim=0, **kwargs):
        self.noise_sim = noise_sim

    def __call__(self, x, theta, noise=None, batch_size=None):
        if noise is None:
            noise = self.noise_sim
        # A simple polynomial simulator of the form \prod_n(x-\theta_n).
        # NOTE: permutation equivariant in roots, hard to learn
        # fn_val = torch.prod(x - theta, dim=-1, keepdim=True)
        # Not permutation invariant, easier
        fn_val = torch.sum(
            theta * x ** torch.arange(theta.shape[-1], device=x.device).view(1, -1),
            dim=-1,
            keepdim=True,
        )
        # Add noise to the output where the noise is a simple function of x
        # fn_val += noise * torch.sin(x * 2 * torch.pi)
        # Return both x and the function value concatenated
        return fn_val.to(x)


class PolySystemSimulator:
    def __init__(self, n_degree: int, device=None):
        self.n_degree = n_degree
        self.device = device

    def poly_from_roots_batch(
        self, x: torch.Tensor, c: torch.Tensor, roots: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate polynomial given roots for batched inputs."""
        # x: (batch_size, n_points)
        # c: (batch_size, 1)
        # roots: (batch_size, n_degree)
        # Output: (batch_size, n_points)

        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # (batch_size, n_points, 1)
        roots_expanded = roots.unsqueeze(1)  # (batch_size, 1, n_degree)

        # Compute (x - roots) for all combinations
        diff = x_expanded - roots_expanded  # (batch_size, n_points, n_degree)

        # Product over roots dimension
        prod = torch.prod(diff, dim=-1)  # (batch_size, n_points)

        return c * prod

    def multiplicative_ripple_batch(self, x, alpha, freq, phase, tau):
        """Multiplicative calibration ripple for batched inputs."""
        return alpha * torch.sin(2 * torch.pi * freq * x + phase) * torch.exp(-x / tau)

    def additive_drift_batch(self, x, sigma_b, ell_b, seed=None):
        """Smooth baseline drift from SE GP for batched inputs - vectorized."""
        if seed is not None:
            torch.manual_seed(seed)

        batch_size, n_points = x.shape

        # Compute all covariance matrices at once
        X = x.unsqueeze(-1)  # (batch_size, n_points, 1)
        X_T = x.unsqueeze(-2)  # (batch_size, 1, n_points)
        dists = (X - X_T) ** 2  # (batch_size, n_points, n_points)

        # Clamp length scales
        ell_b_safe = torch.clamp(ell_b, min=0.01)

        # Build covariance matrices
        K = sigma_b.unsqueeze(-1) ** 2 * torch.exp(
            -0.5 * dists / ell_b_safe.unsqueeze(-1) ** 2
        )

        # Add jitter
        jitter = 1e-4
        eye = (
            torch.eye(n_points, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        )
        K = K + jitter * eye

        # Try batch Cholesky
        try:
            L = torch.linalg.cholesky(K)
        except torch._C._LinAlgError:
            # Fallback: process each matrix individually with eigendecomposition
            L_list = []
            for i in range(batch_size):
                eigenvals, eigenvecs = torch.linalg.eigh(K[i])
                eigenvals = torch.clamp(eigenvals, min=1e-6)
                L_i = eigenvecs @ torch.diag(torch.sqrt(eigenvals))
                L_list.append(L_i)
            L = torch.stack(L_list)

        # Generate random samples
        z = torch.randn(batch_size, n_points, device=x.device)
        drift = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)

        return drift

    def heteroscedastic_std_batch(self, x, sigma0, sigma1):
        """Noise level as a function of x for batched inputs."""
        return torch.sqrt(sigma0**2 + (sigma1 * x) ** 2)

    def __call__(self, x, thetas, nuisance):
        params = torch.cat([nuisance[:, :1], thetas, nuisance[:, 1:]], dim=1)
        return self.simulate(x, params)

    def simulate(
        self, x: torch.Tensor, params: torch.Tensor, drift_seed=None, noiseless=False
    ) -> torch.Tensor:
        """
        Simulate measurements for given x and parameters.

        Args:
            x: Input points, shape (n_points,) or (batch_size, n_points)
            params: Parameters, shape (batch_size, n_params) or (n_params,)
            drift_seed: Seed for drift generation
            noiseless: If True, returns signal without drift or noise

        Returns:
            Measurements, shape (batch_size, n_points) or (n_points,)
        """
        # Handle single parameter set
        if params.ndim == 1:
            params = params.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Handle single x sequence
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(params.shape[0], -1)  # (batch_size, n_points)

        # Extract parameters
        c = params[:, 0:1]  # (batch_size, 1)
        roots = params[:, 1 : self.n_degree + 1]  # (batch_size, n_degree)
        nuis = params[:, self.n_degree + 1 :]  # (batch_size, n_nuis)

        alpha = nuis[:, 0:1]  # (batch_size, 1)
        freq = nuis[:, 1:2]
        phase = nuis[:, 2:3]
        tau = nuis[:, 3:4]
        sigma_b = nuis[:, 4:5]
        ell_b = nuis[:, 5:6]
        sigma0 = nuis[:, 6:7]
        sigma1 = nuis[:, 7:8]

        # True signal - vectorized across batch
        signal = self.poly_from_roots_batch(x, c, roots)  # (batch_size, n_points)

        if noiseless:
            return signal.squeeze(0) if squeeze_output else signal

        # Multiplicative ripple
        ripple_factor = 1.0 + self.multiplicative_ripple_batch(
            x, alpha, freq, phase, tau
        )

        # Additive drift (batch version)
        drift = self.additive_drift_batch(x, sigma_b, ell_b, seed=drift_seed)

        # Heteroscedastic noise
        noise_std = self.heteroscedastic_std_batch(x, sigma0, sigma1)
        noise = noise_std * torch.randn_like(x)

        result = ripple_factor * signal + drift + noise

        return result.squeeze(0) if squeeze_output else result


if __name__ == "__main__":
    # Checking out the new simulator/scanner
    n_degree = 4
    simulator = PolySystemSimulator(n_degree=n_degree)

    # True parameters: first polynomial params, then nuisances
    # fmt: off
    theta_true = torch.tensor([
        1.2,        # c
        0.15, 0.35, 0.58, 0.88,  # roots
        0.07, 6.0, 1.5, 0.9,     # ripple: alpha, freq, phase, tau
        0.04, 0.25,              # GP drift: sigma_b, ell_b
        0.015, 0.06              # heteroscedastic: sigma0, sigma1
    ])

    X0 = torch.linspace(0.0, 1.0, 50)
    # X0 = torch.linspace(0.1, 0.2, 50)
    Y_clean = simulator.simulate(X0, theta_true, noiseless=True)
    # Run the noisy simulator many times to get an average
    Y_average = torch.stack([simulator.simulate(X0, theta_true, drift_seed=i) for i in range(1000)]).mean(dim=0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(X0, simulator.simulate(X0, theta_true), '-', color='gray', alpha=0.5)
    plt.plot(X0, simulator.simulate(X0, theta_true), '-', color='gray', alpha=0.5)
    plt.plot(X0, simulator.simulate(X0, theta_true), '-', color='gray', alpha=0.5)
    plt.plot(X0, Y_average, '-', label='Average Measurements', linewidth=2)
    plt.plot(X0, Y_clean, '-', label='Noiseless Measurements', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simulated Measurements')
    plt.legend()
    plt.grid()
    plt.show()
