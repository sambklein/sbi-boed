"""NLE acquisition strategy - bundles NLE-specific functions."""

from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from nflows.transforms.base import CompositeTransform
from sbi_bax.models.conditional_linear import ConditionalDiagonalLinear

from sbi_bax.experiments.base_acquisition import BaseSequentialAcquisition
from sbi_bax.models.flow import GeneralFlow, StackedFlow, pure_context_inn, spline_inn
from sbi_bax.models.mlp import Mlp
from sbi_bax.utils.sbi import (
    make_norm_layer,
    setup_nle_posterior,
    build_nle_mcmc_posterior,
)
from sbi_bax.utils.torch_utils import no_grad
from sbi_bax.utils.sbi import run_sbi
from sbi_bax.models.flow import DeconditionedFlow

log = logging.getLogger(__name__)


class NleAcquisition(BaseSequentialAcquisition):
    """NLE acquisition that jointly updates posterior and nuisance models."""

    def __init__(
        self,
        *args,
        prior_frac: float = 0.5,
        min_post_frac: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # State (set during __call__)
        self.base_optimizer = None
        self.top_optimizer = None
        # Fraction of samples from prior for training
        self.prior_frac = prior_frac
        self.min_post_frac = min_post_frac

    def _setup_posterior(
        self,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        thetas: torch.Tensor,
        prior,
        n_measured: int,
        data_obs: torch.Tensor,
        measure_dir: Path,
        initial_model,
        sbi_params: dict,
        data,
    ):
        """
        NLE-specific posterior: train likelihood flow and use MCMC.

        Delegates to standalone function for cleaner code organization.
        """
        if self.prior_frac > 0.0:
            # Will sample from prior during MCMC generation, so augment training data accordingly
            n_theta = len(thetas)
            n_prior = int(self.prior_frac * n_theta)
            prior_thetas = prior.sample((n_prior,)).to(thetas.device)
            prior_obs, prior_x = data.build_train_set(
                prior_thetas,
                data_obs,
                n_measured,
            )
            # Concatenate to training data
            n_drop = max(
                int(self.min_post_frac * n_theta), int((1 - self.prior_frac) * n_theta)
            )
            train_obs = torch.cat([train_obs[:n_drop], prior_obs], dim=0)
            train_x = torch.cat([train_x[:n_drop], prior_x], dim=0)
            thetas = torch.cat([thetas[:n_drop], prior_thetas], dim=0)
        return setup_nle_posterior(
            train_obs,
            train_x,
            thetas,
            prior,
            n_measured,
            data_obs,
            measure_dir,
            initial_model,
            sbi_params,
            data,
            self.n_thetas,
            self.device,
        )

    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        thetas: torch.Tensor,
    ):
        """Initialize models and state."""
        self.data = data
        self.data_obs = data_obs.to(self.device)
        self.n_measured = n_measured
        self.n_stepped = 0

        # Load theta samples
        self.theta_estimates = thetas.to(self.device).detach()

        # Build initial training data
        obs, theta_designs, designs = data.train_emulation(
            self.theta_estimates, self.n_mc, data.x_grid.to(self.device)
        )

        # Build base flow p(y | ξ)
        encode_dim = 128
        self.base_flow = GeneralFlow(
            designs.unsqueeze(1),
            obs,
            Mlp(designs.shape[-1], encode_dim),
            lambda dim, context: spline_inn(
                dim,
                128,
                nstack=2,
                context_features=context,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [make_norm_layer(obs, 3.5), ConditionalDiagonalLinear(dim, context)]
                    if self.scale_and_shift
                    else [make_norm_layer(obs, 3.5)]
                ),
            ),
        ).to(self.device)
        self.base_flow.net._embedding_net.fit_norm(designs)

        # Build top flow p(y | θ, ξ)
        def context_slice(context):
            return context[..., -designs.shape[-1] :]

        self.top_flow = StackedFlow(
            theta_designs.unsqueeze(1),
            obs,
            pure_context_inn(  # NOTE: using this function is probably what broke stacking flows here.
                obs.shape[1],
                encode_dim,
                n_layers=2,
                input_transform=CompositeTransform(
                    [
                        make_norm_layer(obs, 3.5),
                        ConditionalDiagonalLinear(obs.shape[1], encode_dim),
                    ]
                    if self.scale_and_shift
                    else [make_norm_layer(obs, 3.5)]
                ),
                # Do all learning in normalized space
                output_transform=make_norm_layer(obs, 3.5, inverse=True),
            ),
            self.base_flow.net,
            Mlp(theta_designs.shape[1], encode_dim),
            base_context_op=context_slice,
        ).to(self.device)
        self.top_flow.net._embedding_net.fit_norm(theta_designs)

        # Setup optimizers
        self.base_optimizer = torch.optim.AdamW(
            self.base_flow.parameters(), lr=self.lr_flow
        )
        self.top_optimizer = torch.optim.AdamW(
            list(self.top_flow.net._transform.parameters())
            + list(self.top_flow.net._embedding_net.parameters()),
            lr=self.lr_flow,
        )

    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None
    ) -> tuple:
        """Generate training data for current designs."""
        n_samples = n_samples or self.n_mc
        return self.data.train_emulation(self.theta_estimates, n_samples, designs)

    def _update_models(self, training_data: tuple) -> dict:
        """Update both flows."""
        # Limit to max batches
        burned_in = self.n_stepped >= self.burn_in
        if not burned_in or self.n_stepped % self.n_model == 0:
            obs, theta_designs, designs = tuple(t.detach() for t in training_data)

            dataset = TensorDataset(obs, theta_designs, designs)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            batch_iter = islice(dataloader, self.max_batches if burned_in else None)

            base_losses = []
            top_losses = []

            for obs_batch, theta_designs_batch, designs_batch in batch_iter:
                # Update base flow
                self.base_optimizer.zero_grad()
                lp_base = self.base_flow.net.log_prob(obs_batch, designs_batch).mean()
                (-lp_base).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.base_flow.parameters(), max_norm=1.0
                )
                self.base_optimizer.step()
                base_losses.append(lp_base.item())

                # Update top flow
                self.top_optimizer.zero_grad()
                with no_grad(self.base_flow):
                    lp_top = self.top_flow.net.log_prob(
                        obs_batch, theta_designs_batch
                    ).mean()
                (-lp_top).backward()
                torch.nn.utils.clip_grad_norm_(self.top_flow.parameters(), max_norm=1.0)
                self.top_optimizer.step()
                top_losses.append(lp_top.item())

            return {"top_flow": top_losses, "base_flow": base_losses}
        return {}

    def _compute_loss(self, designs, n_samples=None, is_final=False):
        """Compute loss (negative EIG) for designs."""
        self.n_stepped += 1
        n_samples = n_samples or self.n_mc
        training_data = self._get_training_data(designs, n_samples=n_samples)
        # Update models if needed
        flow_logs = {}
        if (
            self.n_stepped < self.burn_in or self.n_stepped % self.n_model == 0
        ) and not is_final:
            flow_logs = self._update_models(training_data)

        # Unpack training data
        obs, theta_designs, designs = training_data

        dataset = TensorDataset(obs, theta_designs, designs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * 10 // n_samples * n_samples,
            shuffle=False,
            drop_last=False,
        )
        # Compute EIG batch-wise
        all_eig = []
        for obs_batch, theta_designs_batch, designs_batch in dataloader:
            # Compute the log prob under the top flow
            lp_top = self.top_flow.net.log_prob(obs_batch, theta_designs_batch)
            # Compute the log prob under the base flow
            lp_base = self.base_flow.net.log_prob(obs_batch, designs_batch)
            all_eig.append(lp_top - lp_base)

        # Concatenate and average over MC samples
        eig = torch.cat(all_eig, dim=0).to(designs.device).view(-1, n_samples).mean(-1)
        return -eig, flow_logs


class NleAcquisitionContrastive(NleAcquisition):
    """NLE acquisition that jointly updates posterior and nuisance models."""

    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        thetas: torch.Tensor,
    ):
        """Initialize models and state."""
        self.data = data
        self.data_obs = data_obs.to(self.device)
        self.n_measured = n_measured
        self.n_stepped = 0

        # Load theta samples
        self.theta_estimates = thetas.to(self.device).detach()

        # Build initial training data
        obs, theta_designs, designs = data.train_emulation(
            self.theta_estimates, self.n_mc, data.x_grid.to(self.device)
        )

        # Build base flow p(y | \theta, ξ)
        encode_dim = 128
        self.flow = GeneralFlow(
            theta_designs.unsqueeze(1),
            obs,
            Mlp(theta_designs.shape[-1], encode_dim),
            lambda dim, context: spline_inn(
                dim,
                128,
                nstack=2,
                context_features=context,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [make_norm_layer(obs, 3.5), ConditionalDiagonalLinear(dim, context)]
                    if self.scale_and_shift
                    else [make_norm_layer(obs, 3.5)]
                ),
            ),
        ).to(self.device)
        self.flow.net._embedding_net.fit_norm(theta_designs)

        # Setup optimizers
        self.base_optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.lr_flow)

    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None
    ) -> tuple:
        """Generate training data for current designs."""
        n_samples = n_samples or self.n_mc
        return self.data.train_emulation(self.theta_estimates, n_samples, designs)

    def _update_models(self, training_data: tuple) -> dict:
        """Update both flows."""
        # Limit to max batches
        burned_in = self.n_stepped >= self.burn_in
        if not burned_in or self.n_stepped % self.n_model == 0:
            obs, theta_designs, _ = tuple(t.detach() for t in training_data)

            dataset = TensorDataset(obs, theta_designs)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            batch_iter = islice(dataloader, self.max_batches if burned_in else None)

            loss = []

            for obs_batch, theta_designs_batch in batch_iter:
                # Update base flow
                self.base_optimizer.zero_grad()
                lp_base = self.flow.net.log_prob(obs_batch, theta_designs_batch).mean()
                (-lp_base).backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
                self.base_optimizer.step()
                loss.append(lp_base.item())

            return {"flow": loss}
        return {}

    def _compute_loss(self, designs, n_samples=None, is_final=False):
        """Compute loss (negative EIG) for designs."""
        self.n_stepped += 1
        n_samples = n_samples or self.n_mc
        training_data = self._get_training_data(designs, n_samples=n_samples)
        # Update models if needed
        flow_logs = {}
        if (
            self.n_stepped < self.burn_in or self.n_stepped % self.n_model == 0
        ) and not is_final:
            flow_logs = self._update_models(training_data)

        # Unpack training data
        obs, theta_designs, _ = training_data
        n_contrastive = max(100 // self.n_mc, 10)

        dataset = TensorDataset(obs, theta_designs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * 10 // n_samples * n_samples,
            shuffle=False,
            drop_last=False,
        )
        # Compute EIG batch-wise
        all_eig = []
        for obs_batch, theta_designs_batch in dataloader:
            # Compute the log prob under the base flow
            lp_num = self.flow.net.log_prob(obs_batch, theta_designs_batch)
            lp_contrastive = torch.zeros_like(lp_num)
            # Extract the designs part
            designs_batch = theta_designs_batch[:, self.theta_estimates.shape[1] :]

            for _ in range(n_contrastive):
                # For each sample, draw a random theta
                r_idx = torch.randint(
                    0,
                    len(self.theta_estimates),
                    (len(theta_designs_batch),),
                    device=theta_designs_batch.device,
                )
                theta_contrastive = self.theta_estimates[r_idx]

                # Build a new tensor with contrastive thetas
                theta_designs_contrastive = torch.cat(
                    [theta_contrastive, designs_batch], dim=-1
                )

                # Compute log prob with new tensor
                lp_contrastive += self.flow.net.log_prob(
                    obs_batch, theta_designs_contrastive
                )

            lp_num -= lp_contrastive / (n_contrastive + 1)
            all_eig.append(lp_num)

        # Concatenate and average over MC samples
        eig = torch.cat(all_eig, dim=0).to(designs.device).view(-1, n_samples).mean(-1)
        return -eig, flow_logs


class NleAcquisitionNotStacked(NleAcquisition):
    """NLE acquisition that jointly updates posterior and nuisance models."""

    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        thetas: torch.Tensor,
    ):
        """Initialize models and state."""
        self.data = data
        self.data_obs = data_obs.to(self.device)
        self.n_measured = n_measured
        self.n_stepped = 0

        # Load theta samples
        self.theta_estimates = thetas.to(self.device).detach()

        # Build initial training data
        obs, theta_designs, designs = data.train_emulation(
            self.theta_estimates, self.n_mc, data.x_grid.to(self.device)
        )

        # Build base flow p(y | ξ)
        encode_dim = 128
        self.base_flow = GeneralFlow(
            designs.unsqueeze(1),
            obs,
            Mlp(designs.shape[-1], encode_dim),
            lambda dim, context: spline_inn(
                dim,
                128,
                nstack=2,
                context_features=context,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [make_norm_layer(obs, 3.5), ConditionalDiagonalLinear(dim, context)]
                    if self.scale_and_shift
                    else [make_norm_layer(obs, 3.5)]
                ),
            ),
        ).to(self.device)
        self.base_flow.net._embedding_net.fit_norm(designs)

        # Build top flow p(y | θ, ξ)
        self.top_flow = GeneralFlow(
            theta_designs.unsqueeze(1),
            obs,
            Mlp(theta_designs.shape[-1], encode_dim),
            lambda dim, context: spline_inn(
                dim,
                128,
                nstack=2,
                context_features=context,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [make_norm_layer(obs, 3.5), ConditionalDiagonalLinear(dim, context)]
                    if self.scale_and_shift
                    else [make_norm_layer(obs, 3.5)]
                ),
            ),
        ).to(self.device)
        self.top_flow.net._embedding_net.fit_norm(theta_designs)

        # Setup optimizers
        self.base_optimizer = torch.optim.AdamW(
            self.base_flow.parameters(), lr=self.lr_flow
        )
        self.top_optimizer = torch.optim.AdamW(
            list(self.top_flow.net._transform.parameters())
            + list(self.top_flow.net._embedding_net.parameters()),
            lr=self.lr_flow,
        )


# Crazy code duplication but whatever, is end of the project
class NpeNleAcquisitionContrastive(NleAcquisitionContrastive):
    def _setup_posterior(
        self,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        thetas: torch.Tensor,
        prior,
        n_measured: int,
        data_obs: torch.Tensor,
        measure_dir: Path,
        initial_model,
        sbi_params: dict,
        data,
    ):
        """Setup NPE posterior."""
        data_no_ea = data.combine_no_ea(
            train_obs[:, -1:], train_x[:, -1:], n_measured
        ).squeeze(1)
        # Build the embedding net
        embedding_net = self.context_encoder(input_dim=data_no_ea.shape[-1])
        posterior, theta_estimates = run_sbi(
            data_no_ea,
            thetas,
            prior,
            embedding_net,
            data_obs=data_obs[-1:],
            measure_dir=measure_dir,
            initial_model=initial_model,
            n_samples=self.n_thetas,
            hyperspherical=data.hyperspherical,
            **sbi_params,
        )
        # Wrap in DeconditionedFlow for NPE
        posterior = DeconditionedFlow(
            posterior.posterior_estimator.net.cpu(),
            data_obs[-1:].cpu(),
        )
        return posterior, theta_estimates


# Crazy code duplication but whatever, is end of the project
class NpeNleAcquisition(NleAcquisitionNotStacked):
    def _setup_posterior(
        self,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        thetas: torch.Tensor,
        prior,
        n_measured: int,
        data_obs: torch.Tensor,
        measure_dir: Path,
        initial_model,
        sbi_params: dict,
        data,
    ):
        """Setup NPE posterior."""
        data_no_ea = data.combine_no_ea(
            train_obs[:, -1:], train_x[:, -1:], n_measured
        ).squeeze(1)
        # Build the embedding net
        embedding_net = self.context_encoder(input_dim=data_no_ea.shape[-1])
        posterior, theta_estimates = run_sbi(
            data_no_ea,
            thetas,
            prior,
            embedding_net,
            data_obs=data_obs[-1:],
            measure_dir=measure_dir,
            initial_model=initial_model,
            n_samples=self.n_thetas,
            hyperspherical=data.hyperspherical,
            **sbi_params,
        )
        # Wrap in DeconditionedFlow for NPE
        posterior = DeconditionedFlow(
            posterior.posterior_estimator.net.cpu(),
            data_obs[-1:].cpu(),
        )
        return posterior, theta_estimates


class ExactLikelihood:
    def __init__(self, data):
        # Spoof module API
        self.net = self
        self.data = data

    def to(self, device):
        return self

    def log_prob(self, obs, context):
        # Split theta and design
        theta = context[:, : -self.data.design_dim]
        designs = context[:, -self.data.design_dim :]
        # Simulator provides log prob
        # # Force theta[:, 0] > theta[:, 1] NOTE: sometimes need this
        # mask = theta[:, 0] < theta[:, 1]
        # theta = theta.clone()
        # theta[mask, 0], theta[mask, 1] = theta[mask, 1], theta[mask, 0]
        return self.data.simulator.log_prob(obs, designs, theta, collapse=False)


class NleExactAcquisition(NleAcquisition):
    """If the simulator has a log prob method, use this in place of learned flows."""

    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        thetas: torch.Tensor,
    ):
        """Initialize models and state."""
        self.data = data
        self.data_obs = data_obs.to(self.device)
        self.n_measured = n_measured
        self.n_stepped = 0
        # Defaults because no pretraining needed
        self.dist_penalty = False
        self.burn_in = 0

        # Load theta samples
        self.theta_estimates = thetas.to(self.device).detach()

    def _setup_posterior(
        self,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        thetas: torch.Tensor,
        prior,
        n_measured: int,
        data_obs: torch.Tensor,
        measure_dir: Path,
        initial_model,
        sbi_params: dict,
        data,
    ):
        """Setup exact NLE posterior."""
        # Create working directory
        measure_dir.mkdir(parents=True, exist_ok=True)
        # Build and return MCMC posterior
        return build_nle_mcmc_posterior(
            ExactLikelihood(data),
            prior,
            data,
            n_measured,
            data_obs,
            measure_dir,
            sbi_params,
            self.n_thetas,
        )

    def _compute_loss(self, designs, n_samples=None, is_final=False):
        """Compute loss (negative EIG) for designs."""
        self.n_stepped += 1
        n_samples = n_samples or self.n_mc
        training_data = self._get_training_data(designs, n_samples=n_samples)

        # Unpack training data
        obs, theta_designs, _ = training_data
        n_contrastive = max(100 // self.n_mc, 100)
        # Define a flow model
        self.flow = ExactLikelihood(self.data)
        dataset = TensorDataset(obs, theta_designs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * 10 // n_samples * n_samples,
            shuffle=False,
            drop_last=False,
        )
        # Compute EIG batch-wise
        all_eig = []
        for obs_batch, theta_designs_batch in dataloader:
            # Compute the log prob under the base flow
            lp_num = self.flow.net.log_prob(obs_batch, theta_designs_batch)
            lp_contrastive = torch.zeros_like(lp_num)
            # Extract the designs part
            designs_batch = theta_designs_batch[:, self.theta_estimates.shape[1] :]

            for _ in range(n_contrastive):
                # For each sample, draw a random theta
                r_idx = torch.randint(
                    0,
                    len(self.theta_estimates),
                    (len(theta_designs_batch),),
                    device=theta_designs_batch.device,
                )
                theta_contrastive = self.theta_estimates[r_idx]

                # Build a new tensor with contrastive thetas
                theta_designs_contrastive = torch.cat(
                    [theta_contrastive, designs_batch], dim=-1
                )

                # Compute log prob with new tensor
                lp_contrastive += self.flow.net.log_prob(
                    obs_batch, theta_designs_contrastive
                )

            lp_num -= lp_contrastive / (n_contrastive + 1)
            all_eig.append(lp_num)

        # Concatenate and average over MC samples
        eig = torch.cat(all_eig, dim=0).to(designs.device).view(-1, n_samples).mean(-1)
        # Divide by the mean to stabilize training
        eig = eig / (eig.abs().mean().detach() + 1e-8)
        return -eig, {}


class NpeNleExactAcquisition(NleExactAcquisition):
    """If the simulator has a log prob method, use this in place of learned flows."""

    def _setup_posterior(
        self,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        thetas: torch.Tensor,
        prior,
        n_measured: int,
        data_obs: torch.Tensor,
        measure_dir: Path,
        initial_model,
        sbi_params: dict,
        data,
    ):
        """Setup NPE posterior."""
        data_no_ea = data.combine_no_ea(
            train_obs[:, -1:], train_x[:, -1:], n_measured
        ).squeeze(1)
        # Build the embedding net
        embedding_net = self.context_encoder(input_dim=data_no_ea.shape[-1])
        posterior, theta_estimates = run_sbi(
            data_no_ea,
            thetas,
            prior,
            embedding_net,
            data_obs=data_obs[-1:],
            measure_dir=measure_dir,
            initial_model=initial_model,
            n_samples=self.n_thetas,
            hyperspherical=data.hyperspherical,
            **sbi_params,
        )
        # Wrap in DeconditionedFlow for NPE
        posterior = DeconditionedFlow(
            posterior.posterior_estimator.net.cpu(),
            data_obs[-1:].cpu(),
        )
        return posterior, theta_estimates
