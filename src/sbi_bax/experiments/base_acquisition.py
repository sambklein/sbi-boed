"""Base class for acquisition strategies with common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn
import torch
import numpy as np
import logging
from sbi_bax.experiments.optimizer import optimize_designs
from sbi_bax.utils.torch_utils import get_device
from sbi_bax.utils.sbi import run_sbi
from sbi_bax.models.flow import DeconditionedFlow
from sbi_bax.utils.optim import PairwiseHingeDiversity

log = logging.getLogger(__name__)


class BaseSequentialAcquisition(ABC):
    """Base class for sequential acquisition strategies."""

    def __init__(
        self,
        n_designs: int = 50,
        n_mc: int = 10,
        n_mc_final: int = 100,
        batch_size: int = 256,
        n_steps: int = 500,
        burn_in: int = 50,
        steps_plot: int = 50,
        lr: float = 0.01,
        lr_flow: float = 0.0001,
        max_batches: int = 10,
        n_thetas: int = 10000,
        n_model: int = 10,
        context_encoder: nn.Module | None = None,
        dist_penalty: bool = True,
        scale_and_shift=True,
        save_designs: bool = False,
        anneal_fact: float = 1.0,
    ):
        # Optimization hyperparameters
        self.n_designs = n_designs
        self.n_mc = n_mc
        self.n_mc_final = n_mc_final
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.steps_plot = steps_plot
        self.lr = lr
        self.n_stepped = 0
        self.n_model = n_model
        # Number of thetas to sample as global bank at start of acquisition
        self.n_thetas = n_thetas
        # Model training hyperparameters
        self.lr_flow = lr_flow
        self.batch_size = batch_size
        self.max_batches = max_batches
        # This is the config that configures the context encoder network
        self.context_encoder = context_encoder
        self.device = get_device()
        self.dist_penalty = dist_penalty
        self.scale_and_shift = scale_and_shift
        self.save_designs = save_designs

        # State (set during __call__)
        self.top_flow = None
        self.base_flow = None
        self.theta_estimates = None
        self.data_obs = None
        self.data = None
        self.n_measured = None
        self.anneal_fact = anneal_fact

    def __call__(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        nuisance_estimator,
        n_measured: int,
        thetas: torch.Tensor,
        measure_dir: Path,
        data=None,
    ) -> torch.Tensor:
        """
        Run acquisition to select next design.

        Args:
            data_obs: Current observations
            data_posterior: Trained posterior estimator
            nuisance_estimator: Nuisance parameter estimator (if applicable)
            prior: Prior distribution
            n_measured: Number of measurements taken so far
            thetas: Theta samples from posterior
            train_obs: Training observations
            train_x: Training designs
            measure_dir: Directory for saving results
            data: Data object with simulator methods

        Returns:
            Selected design point
        """
        # Check for checkpoint
        if (measure_dir / "best_x.txt").exists():
            log.info("Loading checkpointed best location.")
            return torch.tensor(
                np.loadtxt(measure_dir / "best_x.txt"), dtype=torch.float32
            )

        # Strategy-specific setup
        self._setup(data_obs, data_posterior, data, n_measured, thetas)

        # Extract bounds from data object if available
        bounds = None
        # If data has bounded domain, extract bounds
        if data is not None and hasattr(data, "data_low") and data.bounded_domain:
            bounds = [torch.as_tensor(data.data_low), torch.as_tensor(data.data_high)]

        # Define distance penalty if requested
        dist_penalty = None
        if self.dist_penalty:
            dist_penalty = PairwiseHingeDiversity(
                min_sep=0.01,
                coef=1_000.0,
                # Subtract once to account for burn in, and again to actually anneal for burn_in steps
                anneal_steps=self.n_steps - (1 + self.anneal_fact) * self.burn_in,
                bounds=bounds,
            )

        # Run optimization using pure function
        # result = optimize_designs_sghmc(
        result = optimize_designs(
            design_set=data.x_grid,
            compute_loss_fn=self._compute_loss,
            step_optimizers=self.step_optimizer,
            n_designs=self.n_designs,
            n_steps=self.n_steps,
            burn_in=self.burn_in,
            lr=self.lr,
            workdir=measure_dir / "warmup",
            steps_plot=self.steps_plot,
            device=self.device,
            eval_samples=self.n_mc_final,
            eig_plot=self._plot_eig,
            bounds=bounds,
            additional_cost=dist_penalty,
            save_final_designs=self.save_designs,
        )

        # Save and return
        np.savetxt(measure_dir / "best_x.txt", result.best_design.numpy())
        return result.best_design

    def step_optimizer(self):
        """Step any internal optimizers if needed."""
        pass

    @abstractmethod
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
        Setup/transform posterior for this acquisition strategy.

        Default implementation: use standard SBI training (NPE case).
        Override in subclass for custom behavior (NLE case).

        Args:
            train_obs: Training observations
            train_x: Training designs
            thetas: Theta samples
            prior: Prior distribution
            n_measured: Current measurement count
            embedding_net: Neural network for embedding observations
            data_obs: Observed data so far
            measure_dir: Directory for saving results
            initial_model: Path to previous model (for warm start)
            sbi_params: SBI training parameters
            data: Data object

        Returns:
            (posterior, theta_samples) tuple
        """
        pass

    def setup_posterior(
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
        posterior, theta_samples = self._setup_posterior(
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
        )
        if hasattr(data, "plot_posterior"):
            data.plot_posterior(
                posterior,
                theta_samples,
                data_obs.cpu(),
                directory=measure_dir,
            )
        return posterior, theta_samples

    @abstractmethod
    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        measure_dir: Path,
    ):
        """
        Initialize models and state for this acquisition strategy.

        Must set:
            - self.data
            - self.theta_estimates
            - Any strategy-specific flows/optimizers
        """
        pass

    @abstractmethod
    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None
    ) -> tuple:
        """Generate training data for current candidate designs."""
        pass

    @abstractmethod
    def _update_models(self, training_data: tuple) -> dict:
        """
        Update flow models for one step.

        Returns:
            Dictionary mapping flow names to lists of losses
        """
        pass

    @abstractmethod
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
        pass

    def _sample_theta_estimates(
        self, data_posterior, n_samples: int = 10000
    ) -> torch.Tensor:
        """Helper to sample theta estimates from posterior."""
        with torch.no_grad():
            theta_estimates = data_posterior.sample((n_samples,)).to(self.device)
            if theta_estimates.dim() == 1:
                theta_estimates = theta_estimates.unsqueeze(-1)
        return theta_estimates.squeeze()

    def _load_theta_estimates(self, measure_dir: Path) -> torch.Tensor:
        """Helper to load pre-computed theta samples."""
        # NOTE: these aren't used?
        return (
            torch.load(measure_dir / "sbi_dt" / "samples_post.pt")
            .to(self.device)
            .detach()
        )

    def _plot_eig(self, designs: torch.Tensor, eig_values: torch.Tensor, figname: Path):
        """Call the data object to plot EIG."""
        if self.data is not None and self.data_obs is not None:
            if hasattr(self.data, "plot_designs_eig"):
                self.data.plot_designs_eig(
                    designs, eig_values.cpu(), self.data_obs.cpu(), figname
                )


class RandomAcquisition(BaseSequentialAcquisition):
    """Random acquisition strategy for benchmarking."""

    def _setup(
        self,
        data_obs: torch.Tensor,
        data_posterior,
        data,
        n_measured: int,
        measure_dir: Path,
    ):
        self.data = data
        self.data_obs = data_obs
        self.n_measured = n_measured
        self.theta_estimates = self._sample_theta_estimates(
            data_posterior, n_samples=self.n_thetas
        )
        # No optimization needed but still run one step for consistency
        self.n_steps = 1

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
        data_no_ea = data.combine_no_ea(train_obs, train_x, n_measured)
        embedding_net = self.context_encoder(input_dim=data_no_ea.shape[-1])
        # Override max epochs from sbi_params
        sbi_params = sbi_params.copy()
        sbi_params["max_epochs"] = 1
        posterior, theta_estimates = run_sbi(
            data_no_ea,
            thetas,
            prior,
            embedding_net,
            data_obs=data_obs,
            measure_dir=measure_dir,
            initial_model=initial_model,
            n_samples=self.n_thetas,
            hyperspherical=data.hyperspherical,
            **sbi_params,
        )
        # Wrap in DeconditionedFlow for NPE
        posterior = DeconditionedFlow(
            posterior.posterior_estimator.net.cpu(),
            data_obs.unsqueeze(0).cpu(),
        )
        return posterior, theta_estimates

    def _get_training_data(
        self, designs: torch.Tensor, n_samples: int | None = None
    ) -> tuple:
        return None  # No training data needed

    def _update_models(self, training_data: tuple) -> dict:
        return {}  # No models to update

    def _compute_loss(
        self,
        designs: torch.Tensor,
        n_samples: int | None = None,
        is_final: bool = False,
    ) -> torch.Tensor:
        """Return random loss for each design."""
        return (
            -torch.rand(
                designs.shape[0], device=designs.device, requires_grad=True
            ).exp()
        ), {}

    # def __call__(self, *args, **kwargs) -> torch.Tensor:
    # TNOTEODO: this skips optimization but also means you can't see the plots optimization creates
    #     """Select random design."""
    #     # Select random design
    #     designs = kwargs.get("data").x_grid
    #     idx = torch.randint(0, designs.shape[0], (1,))
    #     best_design = designs[idx].squeeze()
    #     log.info(f"Random acquisition selected design index {idx.item()}.")
    #     # Save and return
    #     measure_dir = kwargs.get("measure_dir")
    #     measure_dir.mkdir(parents=True, exist_ok=True)
    #     np.savetxt(measure_dir / "best_x.txt", best_design.numpy().reshape(1, -1))
    #     return best_design
