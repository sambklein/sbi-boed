import abc
from pathlib import Path
import numpy as np
import torch

from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from sbi_bax.utils.sbi import run_sbi, train_nuisance_flow
from sbi_bax.utils.torch_utils import get_device, sample_nuisance, sample_theta


# Set up logging
log = logging.getLogger(__name__)


class DummyNuisanceEstimator:
    def sample(self, *args, **kwargs):
        return torch.empty(0)


class BaseAcquisitionExperiment(abc.ABC):
    def __init__(self, cfg: DictConfig, save_config: bool = True):
        """
        Initialize the acquisition experiment with the given configuration.
        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Unpack the configuration so its attributes can be accessed directly
        self.cfg = cfg
        self.save_config = save_config
        self.n_init = self.cfg.measurement.initial_measure
        # Set the random seed for reproducibility
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        # Log the start of the experiment
        log.info(f"Starting experiment: {cfg.exp_name}")

        # Set the device for PyTorch operations everywhere
        self.device = get_device()
        log.info(f"Using device: {self.device}")

        # Initialize the simulator
        self.simulator = hydra.utils.instantiate(
            self.cfg.simulator,
            device=self.device,
        )

        # Initialize the data object
        self.data = hydra.utils.instantiate(self.cfg.data, simulator=self.simulator)
        # Store the true angles for the experiment
        self.true_angles = self.data.true_thetas

        # Initialize the observed data as None
        self.data_obs = None
        # Initialize the class with an embedding network as None
        self.embedding_net = None
        # Store the x_grid for the experiment
        self.x_grid = self.data.x_grid
        # Extract the prior bounds
        self.prior_bounds = list(zip(self.data.prior_low, self.data.prior_high))

    def run(self):
        """
        Run the acquisition experiment.
        This method should be implemented by subclasses to define the specific experiment logic.
        """
        log.info("Starting initial measurement...")
        # Save the resolved configuration to a file
        if self.save_config:
            with open("full_config.yaml", "w") as f:
                OmegaConf.save(config=self.cfg, f=f)
        # Save the true thetas to disk
        torch.save(self.true_angles, "true_angles.pt")
        # Define the prior
        prior = self.data.prior
        # Initialize the SBI posterior with the prior
        current_posterior = deepcopy(prior)
        # Get the initial nuisance parameters estimator
        has_nuisance = hasattr(self.data, "nuisance_prior")
        nuisance_estimator = (
            self.data.nuisance_prior if has_nuisance else DummyNuisanceEstimator()
        )
        # Make the initial measurement
        self.initial_measurement(prior)
        # Run the main acquisition loop
        for n_measured in range(self.n_init, self.cfg.measurement.max_measure + 1):
            # Log the current measurement number
            log.info(
                f"Running measurement {n_measured} of {self.cfg.measurement.max_measure}"
            )
            # Make a subdirectory for this measurement
            measure_dir = Path(f"measurement_{n_measured}")
            measure_dir.mkdir(parents=True, exist_ok=True)

            # Build SBI and flow training datasets
            with torch.no_grad():
                # Sample training thetas
                thetas = sample_theta(
                    current_posterior,
                    measure_dir / "samples.pt",
                    self.cfg.sbi.from_scratch,
                    n_samples=self.cfg.sbi.n_samples,
                )
                # Sample nuisance parameters
                nuisance = sample_nuisance(
                    nuisance_estimator,
                    thetas,
                    self.data_obs,
                    measure_dir / "nuisance_samples.pt",
                    self.cfg.sbi.from_scratch,
                    n_samples=len(thetas),
                    is_first=(n_measured == self.n_init),
                )
                # Update the x_grid using the current posterior
                self.data.update_x_grid(current_posterior, thetas)
                # Make a measurement at every angle for each theta
                train_obs, train_x = self.data.build_train_set(
                    thetas,
                    self.data_obs,
                    n_measured,
                )

            # Combine the dataset without execution path
            data_no_ea = self.data.combine_no_ea(train_obs, train_x, n_measured)
            # Train an SBI model
            log.info("Training SBI model...")
            # Initialize the training with the current posterior
            initial_model = Path(f"measurement_{n_measured - 1}") / "sbi_dt/model.pt"
            if not initial_model.exists():
                initial_model = None
            # Actually train the model
            current_posterior, theta_estimates = self.get_posterior(
                data_no_ea,
                thetas,
                prior,
                n_measured,
                measure_dir=measure_dir / "sbi_dt",
                initial_model=initial_model,
            )
            # Check the posterior
            self.data.check_posterior(
                current_posterior, self.data_obs, directory=measure_dir / "sbi_dt"
            )
            # Update nuisance parameter estimation if needed
            if has_nuisance:
                # Update the nuisance estimator
                nuisance_estimator = train_nuisance_flow(
                    data_no_ea,
                    nuisance,
                    thetas,
                    self.get_embedding_net(),
                    measure_dir=measure_dir / "nuisance_flow",
                    **self.get_sbi_params(n_measured),
                )
                # Check the nuisance estimator
                self.data.check_nuisance(
                    nuisance_estimator,
                    self.data_obs,
                    directory=measure_dir / "nuisance_flow",
                )
                # Update the nuisance samples
                with torch.no_grad():
                    nuisance = nuisance_estimator.sample(
                        torch.Tensor(theta_estimates),
                        self.data_obs.unsqueeze(0).repeat(len(theta_estimates), 1, 1),
                        num_samples=1,
                    ).cpu()
            self.data.check_posterior_predictive(
                self.data_obs,
                theta_estimates,
                nuisance,
                measure_dir / "sbi_dt",
            )
            # Check the posterior predictive on random samples
            self.data.check_posterior_predictive(
                self.data_obs,
                theta_estimates,
                nuisance,
                measure_dir / "sbi_dt",
                random=True,
            )
            # Set the default x for the posterior to the observed angles
            current_posterior.set_default_x(self.data_obs)

            # Only update the posterior after the last measurement, no new data acquired
            if n_measured < self.cfg.measurement.max_measure:
                # Run the acquisition function
                log.info("Running acquisition function...")
                x_next = self.run_acquisition(
                    current_posterior,
                    nuisance_estimator,
                    prior,
                    n_measured,
                    thetas,
                    train_obs,
                    train_x,
                    measure_dir=measure_dir,
                )
                # If x_next is 0dim (1D tensor), convert it to a 1D tensor
                if x_next.dim() == 0:
                    x_next = x_next.unsqueeze(0)
                # Write out the best x value to disk
                np.savetxt(measure_dir / "best_x.txt", x_next.cpu().numpy())

                # Ensure the data_posterior is always conditioned on the observed data
                current_posterior.set_default_x(self.data_obs)

                # Make the next measurement
                _, y_next = self.data.make_measurement(x_next)
                # Build the next observed data
                new_obs = self.data.build_next_obs(y_next, x_next)
                # Append the new measurement to the observed data
                self.data_obs = torch.cat([self.data_obs, new_obs], dim=0)
                # Visualize the next measurement
                self.data.visualize_obs(
                    self.data_obs, measure_dir / f"measurement_{n_measured + 1}.png"
                )
                # Save the observed data to a file
                torch.save(self.data_obs, "data_obs.pt")
        # On experiment completion, write a summary file
        with open("experiment_summary.txt", "w") as f:
            f.write(f"Experiment {self.cfg.exp_name} completed successfully.\n")
            f.write(f"Total measurements taken: {self.cfg.measurement.max_measure}\n")
            f.write(f"Final observed data shape: {self.data_obs.shape}\n")

    def get_posterior(
        self,
        data_no_ea,
        thetas,
        prior,
        n_measured,
        measure_dir,
        initial_model,
    ):
        return run_sbi(
            data_no_ea,
            thetas,
            prior,  # Same prior every time
            self.get_embedding_net(),  # The network used to embed datasets
            data_obs=self.data_obs,
            measure_dir=measure_dir,
            initial_model=initial_model,
            **self.get_sbi_params(n_measured),
        )

    def initial_measurement(self, prior):
        """
        Perform the initial measurement to collect data.
        By default, just measure the first point in the grid.
        """
        # Make an initial measurement randomly (fixed to first in grid)
        indx = (
            0
            if self.n_init == 1
            else np.random.randint(0, len(self.x_grid), size=self.n_init)
        )
        initial_x = self.x_grid[indx]
        # Get the intermediate y (or y itself) from the simulator
        _, self.initial_inter_y = self.data.make_measurement(initial_x)
        # Define the observed data
        self.data_obs = self.data.build_next_obs(self.initial_inter_y, initial_x)
        # Visualize the initial measurement
        self.data.visualize_obs(self.data_obs, "measurement_1.png")

    def get_embedding_net(self):
        # Define the embedding network based on the configuration
        return hydra.utils.instantiate(
            self.cfg.dataset_encoder,
            hydra.utils.instantiate(
                self.cfg.sample_encoder,
                image_shape=self.initial_inter_y.shape,
                scalar_dim=self.x_grid[0].shape[-1],
            ),
        )

    def get_sbi_params(self, n_measured: int) -> dict:
        """
        Get the SBI parameters from the configuration.
        Returns:
            dict: SBI parameters.
        """
        return {
            "sbi_bs": self.cfg.sbi.batch_size,
            "sbi_lr": self.cfg.sbi.lr,
            "max_epochs": self.cfg.sbi.max_epochs,
            "stop_after_epochs": self.cfg.sbi.stop_after_epochs,
            "validation_fraction": self.cfg.sbi.validation_fraction,
            "prior_bounds": self.prior_bounds,
            "simulator": self.simulator,
            "true_angles": self.true_angles,
            "norm_theta": self.cfg.sbi.norm_theta,
            "model": self.cfg.sbi.model,
            "from_scratch": self.cfg.sbi.from_scratch,
        }

    @abc.abstractmethod
    def run_acquisition(
        self,
        data_posterior,
        nuisance_estimator,
        prior,
        n_measured: int,
        thetas: torch.Tensor,
        train_obs: torch.Tensor,
        train_x: torch.Tensor,
        measure_dir: Path,
    ) -> torch.Tensor:
        """
        Run the acquisition function to select the next measurement point.
        Args:
            data_posterior: The posterior distribution p(θ | D_t).
            nuisance_estimator: The nuisance parameter estimator.
            prior: The prior distribution p(θ).
            n_measured (int): The number of measurements taken so far.
            thetas (torch.Tensor): The current parameter samples.
            nuisance (torch.Tensor): The current nuisance parameters.
            train_obs (torch.Tensor): Observed data from previous measurements.
            train_x (torch.Tensor): Input locations of previous measurements.
            measure_dir (Path): Directory to save acquisition results.
        Returns:
            torch.Tensor: The next measurement point.
        """
        raise NotImplementedError("Subclasses must implement run_acquisition method.")
