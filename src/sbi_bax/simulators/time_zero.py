# SB+ BAX for source finding example
import torch
from torch import nn
import numpy as np
from scipy.optimize import curve_fit


def exp_piecewise_constrained(x, y0, tau, x0):
    """C = -y0 is enforced"""
    C = -y0
    y_model = np.full_like(x, y0 + C)  # equals 0
    mask = x >= x0
    y_model[mask] = y0 * np.exp(-(x[mask] - x0) / tau) + C
    return y_model


def fit_exp(
    x, y, x0_guess=0.0, tau_guess=1.0, y0_guess=None, low_bound=None, high_bound=None
):
    x = x.squeeze()
    y = y.squeeze()
    if y0_guess is None:
        y0_guess = np.max(y) - np.min(y)

    p0 = [y0_guess, tau_guess, x0_guess]
    if low_bound is None:
        low_bound = [0, 1e-6, -5]
    if high_bound is None:
        high_bound = [np.inf, 50, 5]
    bounds = (low_bound, high_bound)  # tau > 0
    # Pack into vector
    p0 = np.array([y0_guess, tau_guess, x0_guess], dtype=float)

    # Ensure bounds are arrays of same length
    low_bound = np.array(low_bound, dtype=float)
    high_bound = np.array(high_bound, dtype=float)

    # Clip initial guess to bounds
    p0 = np.clip(p0, low_bound, high_bound)
    params, _ = curve_fit(exp_piecewise_constrained, x, y, p0=p0, bounds=bounds)
    # y0_fit, tau_fit, x0_fit
    return params


def torch_forward(x, y0, tau, x0):
    """C = -y0 is enforced"""
    y_model = torch.zeros_like(x)
    mask = x >= x0
    y_model[mask] = y0[mask] * torch.exp(-(x[mask] - x0[mask]) / tau[mask]) - y0[mask]
    return y_model


class TimeZeroFinding(nn.Module):
    """Time zero finding example"""

    def __init__(self, seed, device=None, tau_min=1, y_min=1):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.tau_min = tau_min
        self.y_min = y_min

    def __call__(self, xi, thetas, nuisance=None, noiseless=False):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.

        Args:
            xi: measurement_location, shape [batch_dim, 1]
            thetas: time zero [batch_dim, 1]
            nuisance: y0 and tau [batch_dim, 2]

        Returns:
            measurements: shape [1]
        """
        # Split up the nuisance params
        y0 = nuisance[:, :1]
        tau = nuisance[:, 1:]
        # Do some simple checks on tau and y0
        y0 = torch.clip(y0, min=self.y_min)
        tau = torch.clip(tau, min=self.tau_min)
        # If the dimensions of y0, tau and thetas don't match xi then broadcast
        n_xi = xi.shape[0]
        if y0.shape[0] != n_xi:
            y0 = y0.repeat(n_xi, 1)
            tau = tau.repeat(n_xi, 1)
            thetas = thetas.repeat(n_xi, 1)
        # Always reshape xi
        xi = xi.view(-1, 1)
        # Call the function
        y_out = torch_forward(xi, y0, tau, thetas)
        # Add noise if required
        if not noiseless:
            y_out += 2 * torch.rand_like(y_out) - 1
        # Convert to torch Tensor and return
        return y_out


def exp2_piecewise(x, A1, tau1, A2, tau2, T0):
    """
    Piecewise double-exponential:
      y(x) = 0,                   x < T0
           = A1*exp(-(x-T0)/tau1) + A2*exp(-(x-T0)/tau2),  x >= T0
    """
    y_model = np.zeros_like(x, dtype=float)
    mask = x >= T0
    y_model[mask] = A1 * np.exp(-(x[mask] - T0) / tau1) + A2 * np.exp(
        -(x[mask] - T0) / tau2
    )
    return y_model


def fit_exp2(
    x,
    y,
    T0_guess=0.0,
    tau1_guess=1.0,
    tau2_guess=5.0,
    A1_guess=None,
    A2_guess=None,
    low_bound=None,
    high_bound=None,
):
    """
    Fit the piecewise double-exponential using scipy.curve_fit with bounds.
    Parameter order: [A1, tau1, A2, tau2, T0]
    """
    x = x.squeeze()
    y = y.squeeze()

    # Simple amplitude guesses if not provided
    if A1_guess is None:
        A1_guess = float(np.maximum(y.max() - y.min(), 1.0))
    if A2_guess is None:
        A2_guess = float(0.5 * A1_guess)

    p0 = np.array([A1_guess, tau1_guess, A2_guess, tau2_guess, T0_guess], dtype=float)

    if low_bound is None:
        # A1, tau1, A2, tau2 > 0; T0 free in [-5, 5] by default
        low_bound = [0.0, 1e-6, 0.0, 1e-6, -5.0]
    if high_bound is None:
        high_bound = [np.inf, 50.0, np.inf, 50.0, 5.0]

    low_bound = np.array(low_bound, dtype=float)
    high_bound = np.array(high_bound, dtype=float)

    p0 = np.clip(p0, low_bound, high_bound)
    params, _ = curve_fit(exp2_piecewise, x, y, p0=p0, bounds=(low_bound, high_bound))
    # Returns [A1, tau1, A2, tau2, T0]
    return params


def torch_forward_exp2(x, A1, tau1, A2, tau2, T0):
    """
    Torch version of the piecewise double-exponential.
    Shapes: x [..., 1], others broadcastable to x
    """
    y_model = torch.zeros_like(x)
    mask = x >= T0
    x_shift = (x - T0).clamp_min(0.0)
    y_model[mask] = (
        A1[mask] * torch.exp(-x_shift[mask] / tau1[mask])
        + A2[mask] * torch.exp(-x_shift[mask] / tau2[mask])
        - (A1 + A2)[mask]
    )
    return y_model  # Enforce y=0 at x=T0


class TimeZeroDoubleExp(nn.Module):
    """
    Simulator for y = A1*exp(-(T - T0)/tau1) + A2*exp(-(T - T0)/tau2), for T >= T0, else 0.
    thetas: T0 (time zero), shape [batch, 1]
    nuisance: [A1, tau1, A2, tau2], shape [batch, 4]
    """

    def __init__(self, seed, device=None, tau_min=1e-3, amp_min=1e-6):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.tau_min = tau_min
        self.amp_min = amp_min

    def __call__(self, xi, thetas, nuisance=None, noiseless=False):
        """
        Args:
            xi: measurement locations T, shape [N, 1]
            thetas: T0, shape [B, 1]
            nuisance: [A1, tau1, A2, tau2], shape [B, 4]
        Returns:
            y: shape [N, 1]
        """
        # Unpack nuisance
        A1 = nuisance[:, :1]
        tau1 = nuisance[:, 1:2]
        A2 = nuisance[:, 2:3]
        tau2 = nuisance[:, 3:4]
        sigma = nuisance[:, 4:]

        # Simple min constraints
        A1 = torch.clip(A1, min=self.amp_min)
        A2 = torch.clip(A2, min=self.amp_min)
        tau1 = torch.clip(tau1, min=self.tau_min)
        tau2 = torch.clip(tau2, min=self.tau_min)

        # Broadcast to xi length as needed
        n_xi = xi.shape[0]
        if A1.shape[0] != n_xi:
            A1 = A1.repeat(n_xi, 1)
            tau1 = tau1.repeat(n_xi, 1)
            A2 = A2.repeat(n_xi, 1)
            tau2 = tau2.repeat(n_xi, 1)
            thetas = thetas.repeat(n_xi, 1)

        xi = xi.view(-1, 1)

        y_out = torch_forward_exp2(xi, A1, tau1, A2, tau2, thetas)
        if not noiseless:
            y_out = y_out + sigma * torch.randn_like(y_out)

        return y_out


if __name__ == "__main__":
    # Simple test
    simulator = TimeZeroDoubleExp(seed=42)
    xi = torch.linspace(-5, 10, 100).view(-1, 1)
    thetas = torch.tensor([[2.0]])
    nuisance = torch.tensor([[4.0, 1.0, 1.0, 5.0, 0.1]])
    y = simulator(xi, thetas, nuisance, noiseless=False)
    import matplotlib.pyplot as plt

    plt.plot(xi.numpy(), y.numpy())
    plt.show()
