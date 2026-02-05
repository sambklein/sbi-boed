import torch
import torch.distributions as dist


class CesPrior(dist.Distribution):
    def __init__(self, basket_dim=3, device=None):
        self.device = device or torch.device("cpu")
        self.basket_dim = basket_dim
        self.p = 4  # dimensionality: rho (1) + alpha (2) + log_u (1)

        # Define priors for each parameter
        # 1. rho - Beta prior on [0,1]
        self.rho_prior = dist.Beta(
            torch.tensor(1.0, device="cpu"), torch.tensor(1.0, device="cpu")
        )

        # 2. alpha - Dirichlet prior (will be transformed via softmax from normal)
        self.alpha_concentration = torch.ones(self.basket_dim, device="cpu")
        self.alpha_prior = dist.Dirichlet(self.alpha_concentration)

        # 3. log_u - Normal prior (log-normal for u)
        self.log_u_prior = dist.Normal(
            torch.tensor(1.0, device="cpu"), torch.tensor(3.0, device="cpu")
        )

    def sample(self, sample_shape=torch.Size()):
        """Sample from the prior distribution.

        Returns:
            Tensor of shape (*sample_shape, 5) containing [rho, alpha1, alpha2, alpha3, log_u]
        """
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)

        # Sample each component on CPU
        rho_samples = self.rho_prior.sample(sample_shape)  # (*sample_shape,)
        # Clip rho to [1e-3, 1 - 1e-3] to avoid numerical issues
        rho_samples = torch.clamp(rho_samples, min=1e-3, max=1 - 1e-3)
        alpha_samples = self.alpha_prior.sample(sample_shape)[
            ..., :2
        ]  # (*sample_shape, 2)
        log_u_samples = self.log_u_prior.sample(sample_shape)  # (*sample_shape,)

        # Concatenate into single tensor
        samples = torch.cat(
            [
                rho_samples.unsqueeze(-1),  # (*sample_shape, 1)
                alpha_samples,  # (*sample_shape, 2)
                log_u_samples.unsqueeze(-1),  # (*sample_shape, 1)
            ],
            dim=-1,
        )  # (*sample_shape, 4)
        return samples.to(self.device)

    def log_prob(self, value):
        """Calculate log probability of samples.

        Args:
            value: Tensor of shape (..., 5) containing [rho, alpha1, alpha2, log_u]

        Returns:
            Tensor of shape (...,) with log probabilities
        """
        # Move to CPU for computation
        value_cpu = value.to("cpu")

        # Extract components
        rho = value_cpu[..., 0]
        alpha = value_cpu[..., 1:3]
        # Infer alpha3 since they must sum to 1
        alpha3 = 1.0 - torch.sum(alpha, dim=-1, keepdim=True)
        alpha = torch.cat([alpha, alpha3], dim=-1)  # (..., 3)
        log_u = value_cpu[..., 3]

        # Compute log probs
        log_p_rho = self.rho_prior.log_prob(rho)
        log_p_alpha = self.alpha_prior.log_prob(alpha)
        log_p_log_u = self.log_u_prior.log_prob(log_u)

        # Sum (independent priors)
        total_log_prob = log_p_rho + log_p_alpha + log_p_log_u

        return total_log_prob.to(value.device)


class CesData:
    def __init__(
        self,
        simulator,
        n_points,
        hyperspherical=False,
    ):
        # Define the observation dimension, this is design + measurement
        self.obs_dim = 1
        self.design_dim = 6
        self.y_dim = 1
        self.theta_dim = 4
        self.simulator = simulator
        self.hyperspherical = hyperspherical
        # Use a uniform base distribution for the flow
        self.uniform_base = True
        # Define the number of points to sample per axis
        self.n_points = n_points
        # Define the prior over thetas, this is fixed by the benchmark setup of iDAD
        self.prior = CesPrior(device=simulator.device)
        # Sample the true thetas from the prior
        self.true_thetas = self.prior.sample().cpu().squeeze()
        # Set 5 sigma bounds on the prior, used for plotting
        self.prior_low = [1e-3] + [0] * 2 + [-3 * 5 + 1]
        self.prior_high = [1 - 1e-3] + [1] * 2 + [3 * 5 + 1]
        # Set bounds on the action space
        self.data_low = [0]
        self.data_high = [100]
        self.bounded_domain = True
        # This is the dimension of the measurement
        self.physical_dim = 1
        self.genx = torch.Generator().manual_seed(42)
        self.x_grid = self.make_x_grid(self.prior, n_points)
        # The first point in the x_grid is measure first
        # This was solved offline to be a good first measurement
        # self.x_grid[0] = torch.tensor([65.0, 67.0, 56.0, 60.0, 69.0, 57.0])
        self.x_grid[0] = torch.tensor(
            [24.9064, 52.0116, 81.8273, 53.4674, 62.4753, 35.4513]
        )

    def prior_constraint(self, theta):
        # Return a mask of valid thetas under the prior
        alpha_sum = theta[..., 1:3].sum(dim=-1)
        return alpha_sum <= (1.0 - 1e-3)

    def make_x_grid(self, posterior, n_points, thetas=None):
        # Use generator to fix randomness across all jobs
        return 100 * torch.rand(n_points, 6, generator=self.genx)

    def update_x_grid(self, posterior, thetas):
        self.x_grid = self.make_x_grid(posterior, self.n_points, thetas=thetas)

    def make_measurement(self, design_params, true_thetas=None):
        if true_thetas is None:
            true_thetas = self.true_thetas
        sim = self.simulator(design_params, true_thetas)
        return sim, sim

    def build_next_obs(self, y, x):
        # Flatten the image and concatenate with x
        return torch.cat([y.view(-1, 1).cpu(), x.view(-1, 6)], dim=1)

    def split_obs(self, observed_data, n_measured):
        # Invert the build_next_obs operation
        x_dim = self.x_grid.shape[1]
        x_measured = observed_data[..., -x_dim:].unsqueeze(0)
        y_measured = observed_data[..., :-x_dim]
        return y_measured, x_measured

    def combine_no_ea(
        self, y_out: torch.Tensor, x_in: torch.Tensor, n_measured: int
    ) -> torch.Tensor:
        return torch.cat([y_out[:, :n_measured].cpu(), x_in[:, :n_measured]], dim=-1)

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
        # NOTE: could generalize many of these data generation functions
        # Get the number of thetas to generate data for
        n_thetas = thetas.shape[0]
        if x_first is None:
            # Generate fake datasets at the observed x values
            x_obs = self.split_obs(data_obs, n_measured)[1]
            x_first = x_obs.repeat(n_thetas, 1, 1)
        # Randomly sample another set of x_values for each theta
        x_sampled = self.x_grid[
            torch.randint(0, self.x_grid.shape[0], (n_thetas,))
        ].unsqueeze(1)
        # Append x_first to each of these sampled x's
        x_fake = torch.cat([x_first, x_sampled], dim=1)
        # Generate the corresponding y values
        y_fake = self.simulator(
            x_fake.view(-1, self.design_dim),
            thetas.repeat_interleave(n_measured + 1, dim=0).view(-1, self.theta_dim),
        ).view(n_thetas, n_measured + 1, 1)
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
        y_fake = self.simulator(x_fake.view(-1, self.design_dim), fake_thetas).view(
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
        pass

    def check_posterior(self, posterior, sample_path, directory=None):
        pass

    def check_posterior_predictive(
        self, data_obs, theta_samples, nuisance, figure_dir, random=False
    ):
        pass

    def plot_designs_eig(self, xi, eig, data_obs, save_path):
        pass
