"""NPE acquisition strategy - bundles NPE-specific functions."""

from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from sbi_bax.experiments.base_acquisition import BaseSequentialAcquisition
from sbi_bax.models.info_nce import ConditionalNCE
from sbi_bax.models.schedulers import LinearWarmupLRScheduler
from sbi_bax.utils.sbi import run_sbi_nre
from sbi_bax.utils.sbi import run_sbi
from sbi_bax.models.flow import DeconditionedFlow

log = logging.getLogger(__name__)


class ContrastiveAcquisition(BaseSequentialAcquisition):
    """NPE acquisition - minimal state container for NPE-specific functions."""

    def __init__(self, *args, prior_frac: float = 0.2, mi_train: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        # State (set during __call__)
        self.mine_optimizer = None
        self.norm_fit = False
        # Fraction of samples from prior for training
        self.prior_frac = prior_frac
        self.mi_train = mi_train

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
        if self.prior_frac > 0.0:
            # Will sample from prior during MCMC generation, so augment training data accordingly
            n_prior = int(self.prior_frac * len(thetas))
            prior_thetas = prior.sample((n_prior,)).to(thetas.device)
            prior_obs, prior_x = data.build_train_set(
                prior_thetas,
                data_obs,
                n_measured,
            )
            # Concatenate to training data
            train_obs = torch.cat([train_obs, prior_obs], dim=0)
            train_x = torch.cat([train_x, prior_x], dim=0)
            thetas = torch.cat([thetas, prior_thetas], dim=0)
        # Build the embedding net
        embedding_net = self.context_encoder(
            theta_dim=thetas.shape[1], obs_dim=train_obs.shape[1]
        )
        # Reset the SBI lr to 1e-3 for NRE
        # Following configuration is a hack for NRE-specific settings
        sbi_params["sbi_lr"] = 1e-3  # NOTE: this isn't a great way to configure this
        sbi_params["sbi_bs"] = 1024  # NOTE: this isn't a great way to configure this
        sbi_params["max_epochs"] = 500  # NOTE: this isn't a great way to configure this

        posterior, theta_estimates = run_sbi_nre(
            train_obs.view(train_obs.shape[0], -1),
            thetas,
            prior,
            embedding_net,
            # Only pass the observations
            data_obs=data.split_obs(data_obs, n_measured)[0].view(1, -1),
            measure_dir=measure_dir,
            initial_model=initial_model,
            n_samples=self.n_thetas,
            **sbi_params,
        )
        return posterior, theta_estimates

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

        # Sample theta buffer
        self.theta_estimates = thetas.to(self.device).squeeze()
        if self.theta_estimates.dim() == 1:
            self.theta_estimates = self.theta_estimates.unsqueeze(-1)

        # Build design distribution model
        self.estimator = ConditionalNCE(
            x_dim=self.theta_estimates.shape[1],
            y_dim=data.y_dim,
            z_dim=data.design_dim,
        ).to(self.device)
        self.norm_fit = False

        # Setup optimizers
        self.optimizer = torch.optim.AdamW(
            self.estimator.parameters(),
            lr=self.lr_flow,
        )
        self.scheduler = LinearWarmupLRScheduler(
            self.optimizer, warmup_steps=min(int(self.burn_in / 10), 100)
        )

    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None, perturb: bool = False
    ) -> tuple:
        """Generate training data for current designs."""
        n_samples = n_samples or self.n_mc
        obs, xi, theta = self.data.train_future_posterior(
            self.theta_estimates,
            designs,
            self.data_obs,
            self.n_measured,
            n_samples,
            perturb_all=perturb,
            combine=False,
        )
        # Reshape to (n_designs, n_samples, dim)
        return (
            theta.view(-1, n_samples, theta.shape[-1]),
            obs[:, -1:].view(-1, n_samples, obs.shape[-1]),
            xi[:, -1:].view(-1, n_samples, designs.shape[-1]),
        )

    def _update_models(self, training_data: tuple) -> dict:
        """Update top flow."""
        theta, obs, designs = tuple(t.detach() for t in training_data)
        burned_in = self.n_stepped >= self.burn_in
        if not self.norm_fit:
            self.estimator.fit_norm(theta, obs, designs)
            self.norm_fit = True
        if not burned_in or self.n_stepped % self.n_model == 0:
            dataset = TensorDataset(theta, obs, designs)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            batch_iter = islice(dataloader, self.max_batches if burned_in else None)

            losses = []
            for theta, obs, designs in batch_iter:
                self.optimizer.zero_grad()
                if self.mi_train:
                    mi = self.estimator.mi_estimate(theta, obs, designs)
                else:
                    mi = self.estimator.train_critic(theta, obs, designs)
                if torch.isnan(mi).any():
                    raise ValueError("NaN detected in MI estimate during training.")
                (-mi).mean().backward()
                torch.nn.utils.clip_grad_norm_(
                    self.estimator.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                losses.append(mi.mean().item())
            self.scheduler.step()

            return {"mine": losses}
        return {}

    def _compute_eig(
        self, designs: torch.Tensor, training_data: tuple, n_samples: int | None = None
    ) -> torch.Tensor:
        """Compute EIG for designs."""
        self.n_stepped += 1
        n_samples = n_samples or self.n_mc
        div_fact = n_samples / self.n_mc

        dataset = TensorDataset(*training_data)
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.batch_size / div_fact),
            shuffle=False,
            drop_last=False,
        )

        all_eig = []
        for theta_batch, obs_batch, designs_batch in dataloader:
            eig = self.estimator.mi_estimate(
                theta_batch.to(self.device),
                obs_batch.to(self.device),
                designs_batch.to(self.device),
            )
            all_eig.append(eig.cpu())

        # Combine all EIGs
        eig = torch.cat(all_eig, dim=0).to(designs.device)
        return eig

    def _compute_loss(
        self,
        designs: torch.Tensor,
        n_samples: int | None = None,
        is_final: bool = False,
    ) -> torch.Tensor:
        """
        Compute loss (negative EIG) for designs.

        Returns:
            Tensor of shape (n_designs,) with negative EIG for each design
        """
        # Get the training data
        # Update models
        training_data = self._get_training_data(designs, n_samples=n_samples)
        step_logs = {} if is_final else self._update_models(training_data)
        # Compute EIG loss
        eig = self._compute_eig(designs, training_data, n_samples=n_samples)
        return -eig, step_logs


class NpeContrastiveAcquisition(ContrastiveAcquisition):
    """NPE acquisition - minimal state container for NPE-specific functions."""

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
        # Check if the data class has a uniform base option
        uniform_base = getattr(data, "uniform_base", False)
        # Check if the data class has a prior constraint option
        data_constraint = getattr(data, "prior_constraint", None)
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
            uniform_base=uniform_base,
            prior_constraint=data_constraint,
            **sbi_params,
        )
        # Wrap in DeconditionedFlow for NPE
        posterior = DeconditionedFlow(
            posterior.posterior_estimator.net.cpu(),
            data_obs[-1:].cpu(),
        )
        return posterior, theta_estimates
