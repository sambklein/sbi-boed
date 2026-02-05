from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributions as dist


class PharmacoPrior(dist.MultivariateNormal):
    def __init__(self, theta_loc=None, theta_covmat=None, device=None):
        # keep requested device for returning samples
        self.device = device or torch.device("cpu")
        self.p = 3

        # move loc & cov to CPU so cholesky (and other linear-algebra ops) run on CPU
        cpu = torch.device("cpu")
        theta_loc = (
            theta_loc if theta_loc is not None else torch.tensor([1.0, 0.1, 20.0]).log()
        ).to(cpu)
        theta_covmat = (
            theta_covmat if theta_covmat is not None else torch.eye(self.p) * 0.05
        ).to(cpu)

        # construct the distribution on CPU (avoids MPS cholesky issue)
        super().__init__(theta_loc, theta_covmat)

    def sample(self, *args, **kwargs):
        # move drawn samples to requested device
        return super().sample(*args, **kwargs).to(self.device)

    def log_prob(self, value):
        # compute log_prob on CPU and move back to value.device
        return super().log_prob(value.to("cpu")).to(value.device)


def idad_thetas(seed: float, device: torch.device) -> torch.Tensor:
    assert 42 <= seed < 58, "Seed must be between 42 and 57 inclusive"
    return torch.tensor(
        [
            [-0.0633, -2.6179, 2.7335],
            [0.3408, -2.4440, 3.3485],
            [-0.1150, -2.6341, 3.2808],
            [-0.0415, -2.0148, 2.8080],
            [-0.1140, -2.4415, 2.9213],
            [0.0181, -2.2975, 3.1080],
            [0.3725, -2.2834, 3.1512],
            [-0.3683, -2.2515, 2.8202],
            [-0.0291, -2.6376, 2.9172],
            [0.2513, -2.6217, 3.0181],
            [-0.0970, -2.5022, 2.8142],
            [-0.0110, -2.2773, 2.5596],
            [-0.2774, -2.1991, 2.9575],
            [0.1295, -2.3256, 2.6769],
            [0.2800, -2.5436, 2.8505],
            [0.1453, -2.3489, 3.0682],
        ],
        device=device,
    )[int(seed - 42)]


class PharmacoKineticData:
    def __init__(
        self,
        simulator,
        n_points,
        seed,
        match_idad=True,
        hyperspherical=False,
    ):
        # Define the observation dimension, this is design + measurement
        self.obs_dim = 2
        self.design_dim = 1
        self.y_dim = 1
        self.theta_dim = 3
        self.simulator = simulator
        self.hyperspherical = hyperspherical
        # Define the number of points to sample per axis
        self.n_points = n_points
        # Define the prior over thetas, this is fixed by the benchmark setup of iDAD
        mean_theta = list(np.log(np.array([1.0, 0.1, 20.0])))
        self.prior = PharmacoPrior(
            theta_loc=torch.Tensor(mean_theta), device=simulator.device
        )
        # self.true_thetas = self.prior.sample().cpu().squeeze()
        # Match iDAD benchmark thetas (only works for seed in 42-57, hacky)
        if match_idad and 42 <= seed < 58:
            self.true_thetas = idad_thetas(seed, simulator.device).cpu().squeeze()
        else:
            self.true_thetas = self.prior.sample().cpu().squeeze()
        # Sample the true thetas from the prior
        std = np.sqrt(0.05).astype(np.float32)
        # Set 5 sigma bounds on the prior, used for plotting
        self.prior_low = [d - std * 5 for d in mean_theta]
        self.prior_high = [d + std * 5 for d in mean_theta]
        # Set bounds on the action space
        self.data_low = [0]
        self.data_high = [24]
        self.bounded_domain = True
        # This is the dimension of the measurement
        self.physical_dim = 1
        self.genx = torch.Generator().manual_seed(42)
        self.x_grid = self.make_x_grid(self.prior, n_points)
        # The first point in the x_grid is measured first
        self.x_grid[0] = 17.56  # Optimal first measurement indp of seed
        self.x_grid[1] = 0.3223  # Optimal second measurement for seed 42
        self.x_grid[2] = 5.3970  # Optimal third measurement for seed 42
        # Create a full set of real measurements and a full grid
        self.full_grid = torch.linspace(0, 24, 10_000).unsqueeze(-1)
        self.y_true = self.simulator(self.full_grid, self.true_thetas).view(-1, 1)
        # Make a random seed generator

    def make_x_grid(self, posterior, n_points, thetas=None):
        # Use generator to fix randomness across all jobs
        return 24.0 * torch.rand(n_points, 1, generator=self.genx)

    def update_x_grid(self, posterior, thetas):
        self.x_grid = self.make_x_grid(posterior, self.n_points, thetas=thetas)

    def make_measurement(self, design_params, true_thetas=None):
        if true_thetas is None:
            true_thetas = self.true_thetas
        sim = self.simulator(design_params, true_thetas)
        return sim, sim

    def build_next_obs(self, y, x):
        # Flatten the image and concatenate with x
        return torch.cat([y.view(-1, 1), x.view(-1, 1)], dim=1)

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

    def build_train_set(
        self,
        thetas,
        data_obs,
        n_measured,
        x_first=None,
    ):
        """Build a training set for the SBI posterior."""
        # Get the number of thetas to generate data for
        n_thetas = thetas.shape[0]
        if x_first is None:
            # Generate fake datasets at the observed x values
            x_obs = self.split_obs(data_obs, n_measured)[1]
            x_first = x_obs.repeat(n_thetas, 1, 1)
        # Generate the corresponding y values
        y_fake = self.simulator(
            x_first.view(-1),
            thetas.repeat_interleave(n_measured, dim=0).view(-1, 3),
        ).view(n_thetas, n_measured, 1)
        return y_fake, x_first

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
        # Repeat theta to match the number of measurements
        fake_thetas = thetas.repeat_interleave(n_measured + 1, 0)
        # Generate fake datasets at the observed x values
        x_obs = self.split_obs(data_obs, n_measured)[1]
        x_first = x_obs.repeat(n_total, 1, 1)
        # Randomly sample another set of x_values for each theta
        x_sampled = designs.repeat_interleave(n_samples, 0)
        # Append x_first to each of these sampled x's
        x_fake = torch.cat([x_first, x_sampled.unsqueeze(1)], dim=1)
        # Generate the corresponding y values
        y_fake = self.simulator(x_fake.view(-1, self.physical_dim), fake_thetas).view(
            n_total, n_measured + 1, 1
        )
        if combine:
            return self.combine_ea(y_fake, x_fake, n_measured).to(
                fake_thetas.device
            ), thetas.view(n_total, -1)
        else:
            return (
                y_fake.to(fake_thetas.device),
                x_fake.to(fake_thetas.device),
                thetas.view(n_total, -1),
            )

    def visualize_obs(self, data_obs, save_path, true_thetas=None):
        y_obs, xi_obs = self.split_obs(data_obs, 0)
        if true_thetas is not None:
            y_true = self.simulator(self.full_grid, true_thetas).view(-1, 1)
        else:
            y_true = self.y_true
        plt.figure()
        plt.plot(self.full_grid.squeeze(), y_true.squeeze(), "o", label="Simulated")
        plt.plot(xi_obs.squeeze(), y_obs.squeeze(), "o", label="Observed")
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def check_posterior(self, posterior, sample_path, directory=None):
        pass

    def check_posterior_predictive(
        self, data_obs, theta_samples, nuisance, figure_dir, random=False
    ):
        pass

    def plot_designs_eig(self, xi, eig, data_obs, save_path):
        y_design = self.simulator(xi, self.true_thetas, noiseless=True).view(-1, 1)
        plt.figure()
        plt.scatter(
            xi.squeeze(),
            y_design.squeeze(),
            label="EIG",
            c=eig.squeeze(),
            s=np.clip(eig * 50, 10, 30),
        )
        cbar = plt.colorbar()
        plt.scatter(
            data_obs[..., -1].squeeze(),
            data_obs[..., 0].squeeze(),
            label="Observed",
            c="red",
            s=50,
            marker="x",
        )
        # Add colorbar
        cbar.set_label("EIG Value")
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
