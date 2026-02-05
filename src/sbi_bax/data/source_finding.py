from matplotlib import pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.distributions as dist
from sbi.analysis import pairplot
from sbi_bax.utils.torch_utils import get_device, lhs
# NOTE: many of these class methods are redundant from other experiments.


class OrderedSourcesPrior(dist.Distribution):
    def __init__(self, n_sources, physical_dim, device, ordering="radial"):
        self.n_sources = n_sources
        self.physical_dim = physical_dim
        self.ordering = ordering
        self.device = device
        self.base_prior = dist.Independent(
            dist.Normal(
                torch.zeros(n_sources * physical_dim).to(device),
                torch.ones(n_sources * physical_dim).to(device),
            ),
            1,
        )
        # Correction factor for ordering: log(n_sources!)
        # Each ordering selects 1 permutation out of n_sources! possible permutations
        self._log_partition = torch.tensor(
            np.math.lgamma(n_sources + 1), device=device
        )  # log(n!)

    def sample(self, sample_shape=()):
        samples = self.base_prior.sample(sample_shape)
        # If not batched then add batch dim
        if len(sample_shape) == 0:
            samples = samples.unsqueeze(0)

        batch_size = samples.shape[0]
        sources = samples.view(batch_size, self.n_sources, self.physical_dim)

        if self.ordering == "radial":
            distances = torch.norm(sources, dim=2)
            sorted_indices = torch.argsort(distances, dim=1)
        elif self.ordering == "x_coordinate":
            sorted_indices = torch.argsort(sources[..., 0], dim=1)
        else:
            raise NotImplementedError("Only radial or x_coordinate supported")

        # Use gather to reorder each batch
        ordered_sources = torch.gather(
            sources,
            1,
            sorted_indices[..., None].expand(-1, -1, self.physical_dim),
        )
        return ordered_sources.view(batch_size, -1)

    def log_prob(self, value):
        """
        Compute log-probability accounting for the ordering constraint.

        For exchangeable base priors, the ordered distribution has density:
            p_ordered(x) = n! * p_base(x) * I[x is ordered]
        where I[x is ordered] is 1 if x satisfies the ordering, 0 otherwise.

        In log-space:
            log p_ordered(x) = log(n!) + log p_base(x)  [if ordered, else -inf]
        """
        # Compute base prior log-prob
        base_lp = self.base_prior.log_prob(value)  # (batch,)

        # Check if samples satisfy ordering constraint
        batch_size = value.shape[0] if value.dim() > 1 else 1
        sources = value.view(batch_size, self.n_sources, self.physical_dim)

        if self.ordering == "radial":
            distances = torch.norm(sources, dim=2)  # (batch, n_sources)
            # Check if distances are sorted (non-decreasing)
            is_ordered = (distances[:, 1:] >= distances[:, :-1]).all(dim=1)
        elif self.ordering == "x_coordinate":
            x_coords = sources[..., 0]  # (batch, n_sources)
            is_ordered = (x_coords[:, 1:] >= x_coords[:, :-1]).all(dim=1)
        else:
            raise NotImplementedError()

        # Add log(n!) for ordered samples, -inf for unordered
        log_prob = torch.where(
            is_ordered,
            base_lp + self._log_partition,
            torch.full_like(base_lp, float("-inf")),
        )
        return log_prob

    def eval_sample(self, *args):
        # Sample from standard Gaussian to match existing implementations
        return self.base_prior.sample(*args)

    def entropy(self):
        """
        Entropy of the ordered distribution.

        For exchangeable priors:
            H[p_ordered] = H[p_base] - log(n!)

        Intuition: ordering removes log(n!) bits of uncertainty
        (choosing 1 permutation out of n!).
        """
        return self.base_prior.entropy() - self._log_partition


class SourceFindingData:
    def __init__(
        self,
        simulator,
        n_points,
        bounded_domain,
        hyperspherical=True,
        random_init=False,
        n_measure_init=2,
    ):
        self.simulator = simulator
        self.hyperspherical = hyperspherical
        self.physical_dim = self.simulator.physical_dim
        self.n_source = self.simulator.n_source
        # Define y dimension
        self.y_dim = 1  # single intensity measurement
        # Define observation dimension
        self.obs_dim = self.physical_dim + self.y_dim
        # Define the total dimension of the flattened theta
        self.theta_dim = simulator.n_source * simulator.physical_dim
        # Define the design dimension
        self.design_dim = self.physical_dim
        # Define the number of points to sample per axis
        self.n_points = n_points
        # Number of initial measurements for short ablations
        self.n_measure_init = n_measure_init
        # Define the prior over thetas, this is fixed by the benchmark setup of iDAD
        self.prior = OrderedSourcesPrior(
            simulator.n_source,
            simulator.physical_dim,
            device=get_device(),
            ordering="radial",
        )
        # Sample the true thetas from the prior
        self.true_thetas = self.prior.sample().cpu().squeeze()
        # Store the true thetas as a flat tensor
        self._true_thetas = self.true_thetas.view(self.n_source, self.physical_dim)
        # Set rough bounds on the prior, used for plotting
        self.prior_low = [-6] * self.theta_dim
        self.prior_high = [6] * self.theta_dim
        # Set bounds on the data space
        self.data_low = [-6] * self.physical_dim
        self.data_high = [6] * self.physical_dim
        self.bounded_domain = bounded_domain
        self.x_grid = self.make_x_grid(self.prior, n_points)
        # The first point in the x_grid is measure first
        # With this prior that should be the origin
        self.x_grid[0] = torch.zeros(self.physical_dim)
        # Create a full set of real measurements
        self.y_true = self.simulator(self.x_grid, self._true_thetas).view(-1, 1)
        # This is used for ablations
        self.random_init = random_init

    def make_x_grid(self, posterior, n_points, thetas=None):
        # Sample from the prior, this is what a random baseline will sample from
        # iDAD implemented this as samples from the prior.
        with torch.no_grad():
            # Sample from the prior
            thetas = self.prior.sample((n_points * 10,)).cpu()  # Get theta samples
            # Reshape to (n_samples, physical_dim)
            thetas = thetas.view(-1, self.physical_dim)
            # Randomly select n_points from these samples
            indices = torch.randperm(thetas.shape[0])[:n_points]
            return thetas[indices]

    def initial_measurement(self, seed):
        # Use a generator with the given seed for reproducibility
        gen = torch.Generator().manual_seed(seed)
        n_measure = self.n_measure_init
        if self.random_init:
            # Sample three points from the prior
            measurement_points = (
                self.prior.sample((3,))
                .view(-1, self.physical_dim)[:n_measure]
                .to(self._true_thetas.device)
            )
        else:
            # Make one measurement at origin
            first_point = torch.zeros((1, self.physical_dim))
            # Sample an additional two points within 0.5 of each source
            shift = (
                torch.randn(self.n_source * n_measure, self.physical_dim, generator=gen)
                * 0.5
            )
            additional_points = (
                self._true_thetas.repeat_interleave(n_measure, dim=0) + shift
            )
            # Randomly select n_measure - 1 of these points
            indices = torch.randperm(self.n_source * n_measure, generator=gen)[
                : n_measure - 1
            ]
            additional_points = additional_points[indices]
            # Concatenate all measurement points
            measurement_points = torch.cat([first_point, additional_points], dim=0)
        # Make measurements
        obs = self.simulator(measurement_points, self._true_thetas)
        # Build the observed data
        data_obs = self.build_next_obs(obs, measurement_points)
        return data_obs

    def update_x_grid(self, posterior, thetas):
        self.x_grid = self.make_x_grid(posterior, self.n_points, thetas=thetas)

    def make_measurement(self, design_params, true_thetas=None):
        if true_thetas is None:
            true_thetas = self._true_thetas
        else:
            true_thetas = true_thetas.view(-1, self.n_source, self.physical_dim)
        # Generate the image using the simulator
        sim = self.simulator(design_params.view(1, -1), true_thetas)
        # Here the simulator has no internal structure.
        return sim, sim

    def build_next_obs(self, y, x):
        # Flatten the image and concatenate with x
        return torch.cat([y.view(-1, 1), x.view(-1, self.physical_dim)], dim=1)

    def split_obs(self, observed_data, n_measured):
        # Invert the build_next_obs operation
        x_dim = self.x_grid.shape[1]
        x_measured = observed_data[..., -x_dim:].unsqueeze(0)
        y_measured = observed_data[..., :-x_dim]
        return y_measured, x_measured

    def combine_no_ea(
        self, y_out: torch.Tensor, x_in: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return torch.cat([y_out[:, :n_measured], x_in[:, :n_measured]], dim=-1)

    def combine_ea(
        self, y_out: torch.Tensor, x_in: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return torch.cat([y_out, x_in], dim=-1)

    def get_min(self, thetas):
        return 0

    def build_train_set(
        self,
        thetas,
        data_obs,
        n_measured,
        noiseless=False,  # Useful for debugging
    ):
        """
        Build a training set for the SBI posterior.
        This will do something similar to alg_path_gen, but will be much faster.
        """
        # Get the number of thetas to generate data for
        n_thetas = thetas.shape[0]
        # Reshape the flattened thetas back into (n_source, physical_dim, physical_dim)
        thetas = thetas.view(n_thetas, self.n_source, self.physical_dim)
        # Generate fake datasets at the observed x values
        x_obs = self.split_obs(data_obs, n_measured)[1]
        x_fake = x_obs.repeat(n_thetas, 1, 1)
        # Generate the corresponding y values
        y_fake = self.simulator(
            x_fake.view(-1, self.physical_dim),
            thetas.repeat(1, n_measured, 1).view(-1, self.n_source, self.physical_dim),
            noiseless=noiseless,
        ).view(n_thetas, n_measured, 1)
        return y_fake, x_fake

    def train_emulation(self, thetas, n_samples, designs=None):
        """Method to produce a dataset for train p(y | theta, xi)"""
        # If designs is None take the x_grid
        if designs is None:
            designs = self.x_grid
        # Repeat the designs to match n_samples
        x_repeated = designs.repeat_interleave(n_samples, 0)
        # For every design sample a theta
        thetas_sampled = thetas[
            torch.randint(0, thetas.shape[0], (x_repeated.shape[0],))
        ]
        # Generate the corresponding y values
        y_fake = self.simulator(x_repeated, thetas_sampled).view(-1, 1)
        # Join theta and the designs by flattening theta
        theta_designs = torch.cat((thetas_sampled, x_repeated), 1)
        return y_fake, theta_designs, x_repeated

    def train_future_posterior(
        self,
        thetas,
        designs,
        data_obs,
        n_measured,
        n_samples,
        perturb_all=False,
        combine=True,
    ):
        """
        Build a training set for the SBI posterior.
        This will do something similar to alg_path_gen, but will be much faster.
        """
        # Get the total number of flattened samples needed
        n_designs = designs.shape[0]
        n_total = n_designs * n_samples
        # Randomly sample n_total from the thetas
        thetas = thetas[torch.randint(0, thetas.shape[0], (n_total,))]
        # Reshape the flattened thetas back into (n_source, physical_dim, physical_dim)
        thetas = thetas.view(n_total, self.n_source, self.physical_dim)
        # Repeat theta to match the number of measurements
        fake_thetas = thetas.repeat(1, n_measured + 1, 1)
        # Generate fake datasets at the observed x values
        x_obs = self.split_obs(data_obs, n_measured)[1]
        x_first = x_obs.repeat(n_total, 1, 1)
        # Randomly sample another set of x_values for each theta
        x_sampled = designs.repeat_interleave(n_samples, 0)
        if perturb_all:
            # Add small gaussian noise to all x values
            x_sampled = x_sampled + 0.1 * torch.randn_like(x_sampled)
        # Append x_first to each of these sampled x's
        x_fake = torch.cat([x_first, x_sampled.unsqueeze(1)], dim=1)
        # Generate the corresponding y values
        y_fake = self.simulator(
            x_fake.view(-1, self.physical_dim),
            fake_thetas.view(-1, self.simulator.n_source, self.physical_dim),
        ).view(n_total, n_measured + 1, 1)
        if combine:
            return self.combine_ea(y_fake, x_fake, n_measured).to(
                fake_thetas.device
            ), thetas.view(n_total, -1)
        else:
            return y_fake, x_fake, thetas.view(n_total, -1)

    def sim_at_theta(self, thetas, designs):
        """
        Generate simulations at fixed thetas and designs
        """
        # Generate the corresponding y values
        y_fake = self.simulator(
            designs.view(-1, self.physical_dim),
            thetas.view(-1, self.simulator.n_source, self.physical_dim),
        ).view(designs.shape[0], 1)
        return self.combine_ea(y_fake, designs, 1).to(thetas.device)

    def visualize_obs(self, data_obs, save_path, true_thetas=None, true_nuisance=None):
        """Visualize the current observed data"""
        if true_thetas is None:
            thetas = self._true_thetas
        else:
            thetas = true_thetas.view(self.n_source, self.physical_dim)
        total_intensity, xi = self.split_obs(data_obs, 1)
        xi = xi.squeeze(0)
        total_intensity = total_intensity.exp()
        plt.figure(figsize=(6, 6))
        plt.scatter(
            xi[:, 0],
            xi[:, 1],
            c=total_intensity,
            s=np.clip(total_intensity * 50, 0, 30),
            cmap="viridis",
            alpha=0.7,
            norm=LogNorm(vmin=total_intensity.min(), vmax=total_intensity.max()),
        )
        plt.colorbar(label="Measurement Value")
        plt.scatter(
            thetas[:, 0],
            thetas[:, 1],
            c="red",
            marker="x",
            s=50,
        )
        # Add a density plot of self.x_grid and self.y_true as a background
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Source Finding Setup")
        # Create custom legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="grey",
                markersize=8,
                label="Detectors",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="red",
                linestyle="None",
                markersize=8,
                label="True Sources",
            ),
        ]
        plt.legend(handles=legend_elements, frameon=False)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_posterior(self, posterior, theta_estimates, data_obs, directory):
        # Compute the radius for each theta estimate
        n_sources = self.simulator.n_source
        radii = torch.norm(
            torch.tensor(theta_estimates).view(-1, n_sources, self.physical_dim), dim=2
        ).cpu()
        # Sample from the prior to get radii distribution
        n_samples = len(theta_estimates)
        prior_samples = self.prior.sample((n_samples,))
        prior_radii = torch.norm(
            prior_samples.view(-1, n_sources, self.physical_dim), dim=2
        ).cpu()
        # Get the x_obs from data_obs
        x_obs = self.split_obs(data_obs, n_measured=1)[1].cpu()
        # Get the radius of each x_obs
        x_radii = torch.norm(x_obs[:, -1]).repeat(n_sources, 1).t()
        # Get the radius of the true sources
        true_radii = torch.norm(self._true_thetas, dim=1).unsqueeze(0).cpu().numpy()
        # Plot the true radii as vertical lines
        plt.figure(figsize=(8, 6))
        # Create a pairplot of the radii
        pairplot(
            [radii.numpy(), prior_radii.numpy()],
            labels=[f"Source {i} Radius" for i in range(n_sources)],
            points=[true_radii, x_radii.numpy()],
            fig_size=(3 * n_sources, 3 * n_sources),
        )
        directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{directory}/posterior_radii.png", dpi=150)
        plt.close()

    def plot_designs_eig(self, xi, eig, data_obs, save_path):
        if self.physical_dim == 2:
            plt.figure(figsize=(6, 6))
            plt.scatter(
                xi[:, 0],
                xi[:, 1],
                c=eig,
                s=np.clip(eig * 50, 10, 30),
                cmap="viridis",
                alpha=0.7,
                # norm=LogNorm(vmin=eig.min(), vmax=eig.max()),
            )
            plt.colorbar(label="EIG")
            plt.scatter(
                self._true_thetas[:, 0],
                self._true_thetas[:, 1],
                c="red",
                marker="x",
                s=50,
            )
            plt.scatter(
                data_obs[:, 1],
                data_obs[:, 2],
                c="orange",
                marker="x",
                s=50,
            )
            # Add a density plot of self.x_grid and self.y_true as a background
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Eig surface")
            # Create custom legend
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="grey",
                    markersize=8,
                    label="Proposed Detectors",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="red",
                    linestyle="None",
                    markersize=8,
                    label="True Sources",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="orange",
                    linestyle="None",
                    markersize=8,
                    label="Positions Measured",
                ),
            ]
            plt.legend(handles=legend_elements, frameon=False)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()


# class SourceFindingDataSmartGrid(SourceFindingData):
#     def make_x_grid(self, posterior, n_points, thetas=None):
#         if thetas is None:
#             return lhs(n_points, self.physical_dim, lb=self.data_low, ub=self.data_high)
#         # Use the posterior to get a grid of x values
#         # Essentially preferentially sample x values where there are likely to be sources
#         with torch.no_grad():
#             # Reshape to (n_samples, physical_dim)
#             thetas = thetas.view(-1, self.physical_dim)
#             # Randomly select n_points from these samples
#             indices = torch.randperm(thetas.shape[0])[:n_points]
#             return thetas[indices]


class SourceFindingDataSmartGrid(SourceFindingData):
    def make_x_grid(self, posterior, n_points, thetas=None):
        if thetas is None:
            return lhs(n_points, self.physical_dim, lb=self.data_low, ub=self.data_high)
        # Use the posterior to get a grid of x values
        # Essentially preferentially sample x values where there are likely to be sources
        with torch.no_grad():
            # Sample from the posterior if it has sample_and_log_prob method
            if hasattr(posterior, "sample_and_log_prob"):
                thetas, log_prob = posterior.sample_and_log_prob(n_points * 10)
                # Sort by log_prob to get the most likely samples
                sorted_indices = torch.argsort(log_prob.squeeze(), descending=True)
                thetas = thetas.squeeze(0)[sorted_indices]
            # Reshape to (n_samples, physical_dim)
            thetas = thetas.view(-1, self.physical_dim)
            # Randomly select n_points from these samples
            indices = torch.randperm(thetas.shape[0])[:n_points]
            return thetas[indices]
