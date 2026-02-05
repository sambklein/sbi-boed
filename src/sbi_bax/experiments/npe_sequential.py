"""NPE acquisition strategy - bundles NPE-specific functions."""

from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from nflows.transforms.base import CompositeTransform
from sbi_bax.experiments.base_acquisition import BaseSequentialAcquisition
from sbi_bax.models.flow import GeneralFlow, StackedFlow, spline_inn
from sbi_bax.models.conditional_linear import ConditionalDiagonalLinear
from sbi_bax.models.schedulers import LinearWarmupLRScheduler
from sbi_bax.utils.sbi import make_norm_layer
from sbi_bax.utils.sbi import run_sbi
from sbi_bax.models.flow import DeconditionedFlow
import numpy as np

log = logging.getLogger(__name__)


class DesignDistribution:
    """Non-parametric design distribution using Gaussian KDE."""

    def __init__(self, bandwidth: float = 0.5):
        """
        Args:
            bandwidth: Bandwidth parameter for Gaussian kernels (similar to std dev)
        """
        self.bandwidth = bandwidth

    def log_prob(self, designs: torch.Tensor, n_mc: int = 10) -> torch.Tensor:
        """
        Compute log probability of each design under KDE formed by all designs.

        Args:
            designs: [n_designs, design_dim] tensor

        Returns:
            log_probs: [n_designs] tensor of log probabilities
        """
        n_designs = designs.shape[0]
        # Take every n_mc-th design to avoid duplicates
        designs = designs[::n_mc]

        # Compute pairwise distances [n_designs, n_designs]
        cdist = torch.cdist(designs, designs)

        # Gaussian kernel: exp(-dist² / (2*h²))
        # Exclude self (diagonal) by setting to 0
        kernel_values = torch.exp(-(cdist**2) / (2 * self.bandwidth**2))
        kernel_values = kernel_values - torch.diag(
            torch.diag(kernel_values)
        )  # Zero out diagonal

        # Sum over all kernels (leave-one-out density estimate)
        density = kernel_values.sum(dim=-1) / (n_designs - 1)

        # Add normalization constant for Gaussian kernel in design_dim dimensions
        design_dim = designs.shape[-1]
        normalization = -0.5 * design_dim * torch.log(
            2 * torch.tensor(torch.pi)
        ) - design_dim * np.log(self.bandwidth)

        # Return log probability
        log_probs = torch.log(density + 1e-10) + normalization

        # Broadcast to match input shape
        log_probs = log_probs.repeat_interleave(n_mc)

        return log_probs


class NpeAcquisition(BaseSequentialAcquisition):
    """NPE acquisition - minimal state container for NPE-specific functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # State (set during __call__)
        self.flow_optimizer = None
        self.norm_fit = False

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

    def get_design_dist(self):
        # Define a dummy class that will always return 1s
        return DesignDistribution()

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

        # Extract base flow
        self.base_flow = data_posterior

        # Sample theta buffer
        self.theta_estimates = thetas.to(self.device).squeeze()
        if self.theta_estimates.dim() == 1:
            self.theta_estimates = self.theta_estimates.unsqueeze(-1)

        # Build design distribution model
        encode_dim = 128
        self.design_flow = self.get_design_dist()

        # Build top flow
        encode_dim = 128
        self.top_flow = StackedFlow(
            torch.zeros(1, self.theta_estimates.shape[1] + 2),
            self.theta_estimates[:1],
            spline_inn(
                self.theta_estimates.shape[1],
                128,
                nstack=2,
                context_features=encode_dim,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [
                        make_norm_layer(self.theta_estimates, 3.5),
                        ConditionalDiagonalLinear(
                            self.theta_estimates.shape[1], encode_dim
                        ),
                    ]
                    if self.scale_and_shift
                    else [make_norm_layer(self.theta_estimates, 3.5)]
                ),
                output_transform=make_norm_layer(
                    self.theta_estimates, 3.5, inverse=True
                ),
            ),
            self.base_flow,
            self.context_encoder(input_dim=data_obs.shape[-1]),
        ).to(self.device)
        self.norm_fit = (
            False  # NOTE: need to improve how you do this, match what is done in NLE!
        )

        # Setup flow optimizers
        self.flow_optimizer = torch.optim.AdamW(
            list(self.top_flow.net._transform.parameters())
            + list(self.top_flow.net._embedding_net.parameters()),
            lr=self.lr_flow,
        )
        self.flow_scheduler = LinearWarmupLRScheduler(
            self.flow_optimizer, warmup_steps=min(int(self.burn_in / 10), 100)
        )

    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None, perturb: bool = False
    ) -> tuple:
        """Generate training data for current designs."""
        n_samples = n_samples or self.n_mc
        return self.data.train_future_posterior(
            self.theta_estimates,
            designs,
            self.data_obs,
            self.n_measured,
            n_samples,
            perturb_all=perturb,
        )

    def _update_models(self, training_data: tuple) -> dict:
        """Update top flow."""
        y_data, theta_data = tuple(t.detach() for t in training_data)
        if not self.norm_fit:
            with torch.no_grad():
                self.top_flow.net._embedding_net.fit_norm(
                    y_data[:, -1:].to(self.device)
                )
                self.norm_fit = True
        burned_in = self.n_stepped >= self.burn_in
        if not burned_in or self.n_stepped % self.n_model == 0:
            dataset = TensorDataset(y_data, theta_data)
            # Get batch size close to a multiple of self.n_mc
            batch_size = self.batch_size // self.n_mc * self.n_mc
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            batch_iter = islice(dataloader, self.max_batches if burned_in else None)

            losses = []
            weights = []
            for y_batch, theta_batch in batch_iter:
                # Grab the designs from the theta batch
                designs = y_batch[:, -1, -self.data.design_dim :].detach()
                lp_design = self.design_flow.log_prob(designs, self.n_mc)
                # Update top flow
                self.flow_optimizer.zero_grad()
                lp_top = self.top_flow.net.log_prob(
                    theta_batch.to(self.device).detach(),
                    y_batch[:, -1:].to(self.device).detach(),
                )
                # Divide by design density to get correct weighting
                weight = torch.exp(-lp_design.detach())
                lp_top = (lp_top * weight).sum() / weight.sum()
                (-lp_top).mean().backward()
                torch.nn.utils.clip_grad_norm_(self.top_flow.parameters(), max_norm=1.0)
                self.flow_optimizer.step()
                losses.append(lp_top.mean().item())
                weights.append(weight.mean().item())
            self.flow_scheduler.step()

            return {"top_flow": losses, "weights": weights}
        return {}

    def _compute_eig(
        self, designs: torch.Tensor, training_data: tuple, n_samples: int | None = None
    ) -> torch.Tensor:
        """Compute EIG for designs."""
        self.n_stepped += 1
        n_samples = n_samples or self.n_mc
        y_designs, thetas = training_data

        dataset = TensorDataset(y_designs.cpu(), thetas.cpu())
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * 10 // n_samples * n_samples,
            shuffle=False,
            drop_last=False,
        )

        all_eig = []
        for y_batch, theta_batch in dataloader:
            lp = self.top_flow.net.log_prob(
                theta_batch.to(self.device), y_batch[:, -1:].to(self.device)
            )
            eig_batch = lp.view(-1, n_samples).mean(dim=-1)
            all_eig.append(eig_batch.cpu())

        # Combine all EIGs
        eig = torch.cat(all_eig, dim=0).to(designs.device)
        # Plot the expected posteriors every 10 steps
        if self.n_stepped % 100 == 0:
            self._plot_expected_posteriors(
                designs, training_data, eig, n_samples=n_samples
            )
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
        # Compute EIG loss, model conditioned on observed data
        eig = self._compute_eig(designs, training_data, n_samples=n_samples)
        return -eig, step_logs

    def _plot_expected_posteriors(
        self,
        designs: torch.Tensor,
        training_data: tuple,
        eig: torch.Tensor,
        n_samples: int,
    ):
        """Plot expected posterior for best design."""
        pass
        # with torch.no_grad():
        #     figdir = Path("design_optimization_plots")
        #     figdir.mkdir(exist_ok=True, parents=True)
        #     # Unpack training data
        #     y_designs, thetas = training_data
        #     # Get best design
        #     for tag in ["best", "expected", "worst"]:
        #         if tag == "best":
        #             best_idx = torch.argmax(eig)
        #         elif tag == "expected":
        #             best_idx = torch.argmin(torch.abs(eig - eig.mean()))
        #         else:  # worst
        #             best_idx = torch.argmin(eig)
        #         test_design = designs[best_idx]
        #         # Get a corresponding observation
        #         indx = torch.where((designs == test_design).all(dim=-1))[0] * n_samples
        #         if len(indx) != 1:
        #             log.info(
        #                 f"Plotting expected posterior for {tag} design {test_design.cpu().numpy()}"
        #             )
        #             indx = indx[0].unsqueeze(0)
        #         y_best = y_designs[indx]
        #         theta_best = thetas[indx]
        #         # Get the prior bounds
        #         prior_bounds = list(zip(self.data.prior_low, self.data.prior_high))
        #         try:
        #             theta_samples = self.top_flow.net.sample(
        #                 10_000,
        #                 context=y_best.to(self.device),
        #             ).cpu()
        #         except RuntimeError as e:
        #             pass
        #         pairplot(
        #             theta_samples.squeeze(),
        #             points=[
        #                 theta_best.cpu().numpy(),
        #                 torch.cat([test_design.detach()] * 2, -1).cpu().numpy(),
        #             ],
        #             limits=prior_bounds,
        #             figsize=(6, 6),
        #         )
        #         plt.savefig(
        #             figdir / f"posterior_{self.n_stepped}_{tag}.png", bbox_inches="tight"
        #         )
        #         plt.close()


class NpeAcquisitionNotStacked(NpeAcquisition):
    """NPE acquisition - but not using a Stacked flow approach."""

    def get_design_dist(self):
        return DesignDistribution()

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

        # Extract base flow
        self.base_flow = data_posterior

        # Sample theta buffer
        self.theta_estimates = thetas.to(self.device).squeeze()
        if self.theta_estimates.dim() == 1:
            self.theta_estimates = self.theta_estimates.unsqueeze(-1)

        # Build design distribution model
        self.design_flow = self.get_design_dist()

        # Build top flow
        self.top_flow = GeneralFlow(
            torch.zeros(1, data_obs.shape[-1]),
            self.theta_estimates[:1],
            self.context_encoder(input_dim=data_obs.shape[-1]),
            lambda dim, context: spline_inn(
                dim,
                128,
                nstack=2,
                context_features=context,
                tails="linear",
                tail_bound=3.5,
                input_transform=CompositeTransform(
                    [
                        make_norm_layer(self.theta_estimates, 3.5),
                        ConditionalDiagonalLinear(dim, context),
                    ]
                    if self.scale_and_shift
                    else [make_norm_layer(self.theta_estimates, 3.5)]
                ),
            ),
        ).to(self.device)
        self.norm_fit = (
            False  # NOTE: need to improve how you do this, match what is done in NLE!
        )

        # Setup flow optimizers
        self.flow_optimizer = torch.optim.AdamW(
            list(self.top_flow.net._transform.parameters())
            + list(self.top_flow.net._embedding_net.parameters()),
            lr=self.lr_flow,
        )
        self.flow_scheduler = LinearWarmupLRScheduler(
            self.flow_optimizer, warmup_steps=min(int(self.burn_in / 10), 100)
        )


class DummyDesignDistribution:
    def log_prob(self, designs: torch.Tensor, n_mc: int = 10) -> torch.Tensor:
        # Return a tensor of zeros with appropriate shape
        log_probs = torch.zeros(designs.shape[0], device=designs.device)
        return log_probs


class NpeAcquisitionNoDesign(NpeAcquisition):
    """NPE acquisition - minimal state container for NPE-specific functions."""

    def get_design_dist(self):
        # Define a dummy class that will always return 1s
        return DummyDesignDistribution()


class NpeAcquisitionNoDesignNotStacked(NpeAcquisitionNotStacked):
    """NPE acquisition - minimal state container for NPE-specific functions."""

    def get_design_dist(self):
        # Define a dummy class that will always return 1s
        return DummyDesignDistribution()
