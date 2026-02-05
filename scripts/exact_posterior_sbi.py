# Look at examples where the exact posterior is known and compare with SBI estimates
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
import typer

from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    process_prior,
)


class Simulator:
    def __init__(self, std):
        self.std = std

    def mean(self, x, theta):
        return theta * x

    def sample(self, x, theta):
        return np.random.normal(loc=self.mean(x, theta), scale=self.std, size=x.shape)

    def likelihood(self, x, y, theta):
        return norm.pdf(y, loc=self.mean(x, theta), scale=self.std)


class ThetaPrior:
    def __init__(self, limits: list):
        self.limits = limits

    def sample(self, n_samples):
        return np.random.uniform(self.limits[0], self.limits[1], n_samples)

    def likelihood(self, theta):
        if self.limits[0] <= theta <= self.limits[1]:
            return 1 / (self.limits[1] - self.limits[0])
        else:
            return 0


def unnormed_exact_posterior(
    x: np.ndarray,
    y: np.ndarray,
    theta: float,
    simulator: callable,
    theta_prior: callable,
) -> np.ndarray:
    """
    Computes the exact posterior distribution for a given parameter theta.
    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Observed data.
        theta (float): Parameter of the model.
        simulator (function): Function to simulate data.
        theta_prior (function): Prior distribution function for theta.
    Returns:
        posterior (np.ndarray): Posterior distribution values.
    """
    # Get the likelihood across all x/y pairs for the given theta
    pdf = np.prod(simulator.likelihood(x, y, theta))
    # Compute the likelihood for the given theta
    theta_prior = theta_prior.likelihood(theta)
    return pdf * theta_prior


def main(
    plot_dir: Path = "figures/exact_posterior_sbi",
    seed: int = 42,
    std: float = 0.1,
    theta_true: float = 0.5,
    prior_min: float = 0.25,
    prior_max: float = 0.75,
    n_theta: int = 100_000,
    n_theta_sbi: int = 5_000,
):
    """
    Compare exact posterior with SBI estimates for a simple linear model.

    This script is intended to look at p(theta | D_t, e_A).
    An approximation using Bayes' rule will be compared to drawing samples
    using the proposed sampling procedure.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Define the simulator and prior
    simulator = Simulator(std)
    theta_prior = ThetaPrior(limits=[prior_min, prior_max])

    # Discretize the input space
    x_vals = np.linspace(-2, 2, 10)
    # Generate some observed data, will only ever use one point
    obs_x = np.random.choice(x_vals, size=1)
    obs_y = simulator.sample(obs_x, theta_true)
    # Generate an execution path as the x_vals repeated 2 times
    exec_path_x = np.tile(x_vals, 2)
    exec_path_y = simulator.sample(exec_path_x, theta_true)

    # Compute the exact posterior for a range of theta values
    theta_values = np.linspace(prior_min, prior_max, n_theta)
    posterior_values = np.array(
        [
            unnormed_exact_posterior(obs_x, obs_y, theta, simulator, theta_prior)
            for theta in theta_values
        ]
    )

    # Normalize the posterior values using trapezoidal rule
    norm_fact = np.trapz(posterior_values, theta_values)
    posterior_values /= norm_fact

    # Compute an estimate of the posterior using our exact knowledge and Bayes' rule
    # p(\theta | D_t, e_a) \proportional_to p(e_A | \theta)p(\theta | D_t)
    # We already have p(\theta | D_t) for theta_values from the posterior values computed above
    # Get the likelihood of exec_path_y given exec_path_x and theta_values
    exec_path_likelihood = np.prod(
        np.array(
            [
                simulator.likelihood(exec_path_x, exec_path_y, theta)
                for theta in theta_values
            ]
        ),
        1,
    )
    # Compute the likelihood of the execution path given theta
    theta_likelihood = posterior_values * exec_path_likelihood
    # Normalize the likelihood
    theta_likelihood /= np.trapz(theta_likelihood, theta_values)

    # Set up an SBI estimate of the posterior
    # Sample a theta_tilde using our current posterior
    t_idx = np.random.choice(
        np.arange(n_theta),
        size=n_theta_sbi,
        p=posterior_values / np.sum(posterior_values),
    )
    theta_tilde = theta_values[t_idx]
    # Sample a faux dataset using the sampled theta_tilde
    sim_y = simulator.sample(obs_x.repeat(len(theta_tilde)), theta_tilde)
    # For every dataset in this sample, compute the posterior and draw a sample from it
    theta_sbi_samples = []
    theta_values_int = np.linspace(prior_min, prior_max, n_theta_sbi)
    for i in range(len(theta_tilde)):
        # Compute the posterior pretending that theta_tilde[i] is the true parameter (condition on sim_y[i])
        internal_posterior = np.array(
            [
                unnormed_exact_posterior(obs_x, sim_y[i], theta, simulator, theta_prior)
                for theta in theta_values_int
            ]
        )
        # Sample from this posterior
        sample = np.random.choice(
            theta_values_int, p=internal_posterior / np.sum(internal_posterior)
        )
        theta_sbi_samples.append(sample)
    # For each of these samples generate an execution path
    theta_sbi_samples = torch.tensor(np.array(theta_sbi_samples), dtype=torch.float32)
    exec_path_sbi_y = np.array(
        [simulator.sample(exec_path_x, tt) for tt in theta_sbi_samples]
    )

    # Now train an SBI model on the execution path and faux dataset (order matters so no x's needed.)
    condition = torch.tensor(
        np.concatenate([exec_path_sbi_y, sim_y.reshape(-1, 1)], axis=1),
        dtype=torch.float32,
    )
    # Define the prior for SBI
    prior = BoxUniform(low=torch.tensor([prior_min]), high=torch.tensor([prior_max]))
    # Check prior, return PyTorch prior.
    prior, _, _ = process_prior(prior)
    # Make an inference model
    inference = NPE(prior=prior)
    inference = inference.append_simulations(theta_sbi_samples.view(-1, 1), condition)
    # Train the inference model
    density_estimator = inference.train()
    # Build the posterior from the trained density estimator
    posterior = inference.build_posterior(density_estimator)
    # Set the default x for the posterior to the true exec_path ys and obs_y
    true_condition = torch.tensor(
        np.concatenate([exec_path_y.reshape(1, -1), obs_y.reshape(-1, 1)], axis=1),
        dtype=torch.float32,
    )
    posterior.set_default_x(true_condition)
    # Sample from the posterior
    with torch.no_grad():
        sbi_samples = posterior.sample((n_theta,))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, theta_likelihood, label="Exact Posterior", linewidth=2)
    plt.hist(
        sbi_samples.numpy(),
        bins=30,
        density=True,
        alpha=0.5,
        label="SBI Samples",
        color="orange",
        histtype="step",
    )
    plt.axvline(theta_true, color="red", linestyle="--", label=f"True θ = {theta_true}")
    plt.xlabel("θ")
    plt.ylabel("Posterior Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Save the plot
    plot_file = plot_dir / "exact_posterior.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    typer.echo(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    typer.run(main)
