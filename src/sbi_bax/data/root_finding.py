from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from contextlib import redirect_stdout, redirect_stderr
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

from sbi_bax.models.flow import NuisanceFlow
from sbi_bax.utils.torch_utils import get_device


class RootFindingData:
    def __init__(
        self,
        simulator,
        prior_bounds,
        prior_nuisance_bounds,
        data_bounds,
        n_points,
        true_nuisance: torch.Tensor,
        n_scan: int,
        true_thetas,
    ):
        self.simulator = simulator
        self.prior_bounds = prior_bounds
        self.data_bounds = data_bounds
        self.n_points = n_points
        self.true_thetas = true_thetas
        self.n_dim = len(true_thetas)
        self.true_nuisance = torch.tensor(true_nuisance).view(1, -1)
        self.n_nuisance = len(true_nuisance)
        # Store the scan definition
        self.n_scan = n_scan
        # Build the prior over thetas
        self.prior = self.make_prior(prior_bounds)
        # Build the prior over the nuisance parameters
        self.nuisance_prior = self.make_nuisance_prior(prior_nuisance_bounds)
        self.prior_low = [prior_bounds[0]] * self.n_dim
        self.prior_high = [prior_bounds[1]] * self.n_dim
        self.data_low = [data_bounds[0]] * self.n_dim
        self.data_high = [data_bounds[1]] * self.n_dim
        self.x_grid = self.make_x_grid(data_bounds, n_points)
        # Create a full set of real measurements with no noise
        self.x_true = self.x_grid
        self.y_true = self.simulator.simulate(
            self.x_true,
            self.full_params(self.true_thetas.view(1, -1), self.true_nuisance),
            noiseless=True,
        ).view(-1, 1)

    def full_params(self, thetas, nuisance):
        return torch.cat([nuisance[:, :1], thetas, nuisance[:, 1:]], dim=1)

    def split_out_nuisance(self, thetas_and_nuisance):
        thetas = thetas_and_nuisance[: self.n_dim]
        nuisance = thetas_and_nuisance[self.n_dim :]
        return thetas, nuisance

    def make_prior(self, prior_bounds):
        """
        Create a prior distribution over the x_in.
        Args:
            prior_bounds (list): List of two elements defining the bounds for the prior.
        Returns:
            BoxUniform: A uniform prior distribution over the x_in.
        """
        return BoxUniform(
            low=torch.tensor([prior_bounds[0]] * self.n_dim, dtype=torch.float32),
            high=torch.tensor([prior_bounds[1]] * self.n_dim, dtype=torch.float32),
            device=get_device(),
        )

    def make_nuisance_prior(self, prior_bounds_dict):
        """
        Create a prior distribution over the nuisance parameters with different bounds per parameter.
        Args:
            prior_bounds_dict (dict): Dictionary with bounds for each parameter type.
        Returns:
            BoxUniform: A uniform prior distribution over the nuisance parameters.
        """
        # Extract bounds for each parameter type
        c_bounds = prior_bounds_dict.get("c", [-2.0, 2.0])
        alpha_bounds = prior_bounds_dict.get("alpha", [-0.5, 0.5])
        freq_bounds = prior_bounds_dict.get("freq", [0.5, 10.0])
        phase_bounds = prior_bounds_dict.get("phase", [0.0, 2 * np.pi])
        tau_bounds = prior_bounds_dict.get("tau", [0.1, 5.0])
        sigma_b_bounds = prior_bounds_dict.get("sigma_b", [0.01, 1.0])
        ell_b_bounds = prior_bounds_dict.get("ell_b", [0.05, 2.0])
        sigma0_bounds = prior_bounds_dict.get("sigma0", [0.001, 0.1])
        sigma1_bounds = prior_bounds_dict.get("sigma1", [0.0, 0.05])

        # Build concatenated bounds
        low_bounds = (
            [c_bounds[0]]  # coefficient
            + [
                alpha_bounds[0],
                freq_bounds[0],
                phase_bounds[0],
                tau_bounds[0],
            ]  # ripple params
            + [sigma_b_bounds[0], ell_b_bounds[0]]  # GP params
            + [sigma0_bounds[0], sigma1_bounds[0]]  # noise params
        )

        high_bounds = (
            [c_bounds[1]]  # coefficient
            + [
                alpha_bounds[1],
                freq_bounds[1],
                phase_bounds[1],
                tau_bounds[1],
            ]  # ripple params
            + [sigma_b_bounds[1], ell_b_bounds[1]]  # GP params
            + [sigma0_bounds[1], sigma1_bounds[1]]  # noise params
        )

        return BoxUniform(
            low=torch.tensor(low_bounds, dtype=torch.float32),
            high=torch.tensor(high_bounds, dtype=torch.float32),
            device=get_device(),
        )

    def make_x_grid(self, data_bounds, n_points):
        # The possible x's here will be all sequential points within bounds of length n_scan
        scan_points = (
            torch.linspace(data_bounds[0], data_bounds[1], n_points + self.n_scan - 1)[
                : self.n_scan
            ]
            - data_bounds[0]
        )
        # scan_points = scan_points[[0, -1]]
        return torch.linspace(
            data_bounds[0], data_bounds[1] - scan_points[-1], n_points
        ).view(-1, 1) + scan_points.view(1, -1)

    def make_measurement(self, design_params):
        # Generate the scan parameters
        x_scan = torch.linspace(design_params[0], design_params[1], self.n_scan)
        # Generate the image using the simulator
        sim = self.simulator(x_scan, self.true_thetas.view(1, -1), self.true_nuisance)
        # Here the simulator has no internal structure.
        return sim, sim

    def build_next_obs(self, image, x):
        # Flatten the image and concatenate with x
        return torch.cat([image.view(1, -1), x.view(1, -1)], dim=1)

    def split_obs(self, observed_data, n_measured):
        # Invert the build_next_obs operation
        x_dim = self.x_grid.shape[1]
        x_measured = observed_data[:, -x_dim:].unsqueeze(0)
        y_measured = observed_data[:, :-x_dim]
        return y_measured, x_measured

    def combine_no_ea(
        self, y_out: torch.Tensor, x_in: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return torch.cat([y_out[:, :n_measured], x_in[:, :n_measured]], dim=-1)

    def combine_ea(
        self, y_out: torch.Tensor, x_in: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return torch.cat([y_out, x_in], dim=-1)

    def build_train_set(
        self,
        thetas,
        nuisance,
        posterior,
        data_obs,
        n_measured,
    ):
        """
        Build a training set for the SBI posterior.
        This will do something similar to alg_path_gen, but will be much faster.
        """
        n_thetas = thetas.shape[0]
        # Generate fake datasets at the observed x values
        x_obs = self.split_obs(data_obs, n_measured)[1]
        # Randomly sample another set of x_values for each theta
        x_sampled = self.x_grid[
            torch.randint(0, self.x_grid.shape[0], (n_thetas,))
        ].unsqueeze(1)
        # Append an x_obs to each of these sampled x's
        x_fake = torch.cat([x_obs.repeat(n_thetas, 1, 1), x_sampled], dim=1)
        # Generate the corresponding y values
        y_fake = self.simulator(
            x_fake.view(-1, self.n_scan),
            thetas.repeat_interleave(n_measured + 1, dim=0),
            nuisance.repeat_interleave(n_measured + 1, dim=0),
        ).view(n_thetas, n_measured + 1, self.n_scan)
        return y_fake, x_fake

    def visualize_obs(self, data_obs, save_path):
        pass

    def check_posterior(self, posterior, sample_path, directory=None):
        posterior.set_default_x(sample_path)
        # Generate samples from the posterior
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                thetas = posterior.sample((500,)).cpu()
            thetas_duplicated = thetas.repeat_interleave(self.x_grid.shape[0], dim=0)
        # Simulate the observations for these thetas
        y_out = self.simulator(
            self.x_grid.repeat(thetas.shape[0], 1),
            thetas_duplicated,
            self.true_nuisance.repeat(thetas_duplicated.shape[0], 1),
        ).view(thetas.shape[0], -1)  # NOTE: is this external reshape correct?
        # Plot the posterior samples
        fig, ax = plt.subplots()

        # Compute quantiles across posterior samples at each x point
        x_grid_np = self.x_grid.cpu().numpy().flatten()
        y_out_np = y_out.cpu().numpy()

        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        quantiles = np.percentile(y_out_np, percentiles, axis=0)

        # Plot quantile bands
        ax.fill_between(
            x_grid_np,
            quantiles[0],
            quantiles[4],
            alpha=0.2,
            color="blue",
            label="90% CI",
        )
        ax.fill_between(
            x_grid_np,
            quantiles[1],
            quantiles[3],
            alpha=0.4,
            color="blue",
            label="50% CI",
        )
        ax.plot(
            x_grid_np, quantiles[2], color="blue", linewidth=2, label="Posterior Median"
        )

        # Plot true data
        ax.scatter(
            self.x_true.cpu().numpy(),
            self.y_true.cpu().numpy(),
            color="red",
            label="True Data",
            s=20,
            zorder=10,
            alpha=0.8,
        )

        # Add the sample path
        ax.scatter(
            sample_path[..., 1].cpu().numpy(),
            sample_path[..., 0].cpu().numpy(),
            color="green",
            label="Sample Path",
            s=20,
            zorder=10,
            alpha=0.8,
        )

        ax.legend(frameon=False)
        ax.set_title("Posterior Credible Intervals vs True Data")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if directory is not None:
            fig.savefig(
                directory / "posterior_samples.png", dpi=150, bbox_inches="tight"
            )
        plt.close(fig)

    def check_nuisance(
        self, nuisance_estimator: NuisanceFlow, data_obs, directory=None
    ):
        # Evaluate the nuisance estimator on the observed data and given thetas
        with torch.no_grad():
            nuisance_samples = nuisance_estimator.sample(
                self.true_thetas.unsqueeze(0), data_obs.unsqueeze(0), num_samples=10_000
            )
        nuisance_samples = nuisance_samples.cpu().numpy().squeeze()
        pairplot(
            nuisance_samples, points=self.true_nuisance.cpu().numpy(), figsize=(10, 10)
        )
        plt.savefig(directory / "nuisance_samples.png", dpi=150, bbox_inches="tight")
        plt.close()
