"""Functional experiment orchestration with explicit data flow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import torch
from omegaconf import DictConfig
from sbi_bax.experiments.base_acquisition import BaseSequentialAcquisition
from sbi_bax.utils.sbi import train_nuisance_flow
import logging

# Set up logging
log = logging.getLogger(__name__)


@dataclass
class ExperimentState:
    """Immutable state snapshot at a given measurement round."""

    n_measured: int
    data_obs: torch.Tensor
    posterior: any  # or a Protocol
    nuisance_estimator: any
    theta_samples: torch.Tensor
    nuisance_samples: torch.Tensor
    x_grid: torch.Tensor


def make_initial_measurement(
    data, x_grid: torch.Tensor, n_init: int, seed: int
) -> torch.Tensor:
    """Make initial measurement(s) and return observed data."""
    torch.manual_seed(seed)
    # Take the first n_init points from the grid
    initial_x = x_grid[:n_init]
    _, initial_y = data.make_measurement(initial_x)
    return data.build_next_obs(initial_y, initial_x)


def make_measurement(
    data, x_next: torch.Tensor, data_obs: torch.Tensor
) -> torch.Tensor:
    """Execute measurement and return updated observations."""
    _, y_next = data.make_measurement(x_next)
    obs_next = data.build_next_obs(y_next, x_next)
    return torch.cat([data_obs, obs_next], dim=0)


def run_experiment_round(
    state: ExperimentState,
    cfg: DictConfig,
    data,
    prior,
    acquisition_fn: BaseSequentialAcquisition,
    measure_dir: Path,
) -> ExperimentState:
    """
    Run one round of the experiment loop.

    Returns:
        (new_state, measurement_result)
        measurement_result is None if this is the final round.
    """
    # 1. Grab current number of measurements
    n_measured = state.n_measured

    # Check if we're done (safe exit, shouldn't be here but is what it is)
    if n_measured >= cfg.measurement.max_measure:
        # Final round, no acquisition
        return ExperimentState(
            n_measured=n_measured,
            data_obs=state.data_obs,
            posterior=state.posterior,
            nuisance_estimator=state.nuisance_estimator,
            theta_samples=state.theta_samples,
            nuisance_samples=state.nuisance_samples,
            x_grid=state.x_grid,
        )

    # 2. Build training dataset
    with torch.no_grad():
        # Get training data from data object
        train_obs, train_xi = data.build_train_set(
            state.theta_samples,
            state.data_obs,
            n_measured,
        )

    # 3. Train posterior
    initial_model = None  # Don't do warm starts.
    # Get SBI training parameters
    sbi_params = get_sbi_params(cfg, data)
    # NOTE: should pass this as a separate function.
    posterior, theta_estimates = acquisition_fn.setup_posterior(
        train_obs,
        train_xi,
        state.theta_samples,
        prior,
        n_measured,
        state.data_obs,
        measure_dir / "sbi_dt",
        initial_model,
        sbi_params,
        data,
    )
    # Update x_grid in data object based on new posterior samples
    data.update_x_grid(posterior, theta_estimates)

    # 4. Train nuisance model if needed
    has_nuisance = hasattr(data, "nuisance_prior")
    if has_nuisance:
        nuisance_estimator = train_nuisance_flow(
            data.combine_no_ea(train_obs, train_xi, n_measured),
            state.nuisance_samples,
            state.theta_samples,
            embeddor=None,  # NOTE: this won't work. Maybe don't support nuisance flows anymore? Or just configure properly...
            measure_dir=measure_dir,
            **sbi_params,
        )
        # Resample nuisance with updated model
        with torch.no_grad():
            nuisance = nuisance_estimator.sample(
                theta_estimates,
                state.data_obs.unsqueeze(0).repeat(len(theta_estimates), 1, 1),
                num_samples=1,
            ).cpu()
    else:
        nuisance_estimator = state.nuisance_estimator
        nuisance = state.nuisance_samples

    # 6. Run acquisition
    x_next = acquisition_fn(
        data_obs=state.data_obs,
        data_posterior=posterior,
        nuisance_estimator=state.nuisance_estimator,
        n_measured=state.n_measured,
        thetas=theta_estimates,
        measure_dir=measure_dir,
        data=data,  # Pass data to acquisition function
    )

    # 7. Make measurement
    data_file = measure_dir / "data_obs.pt"
    if cfg.measurement.from_scratch or cfg.sbi.from_scratch or not data_file.exists():
        obs_next = make_measurement(data, x_next, state.data_obs)
        # Save measurement
        log.info("Saving new measurement data to disk.")
        torch.save(obs_next, data_file)
    else:
        # Load measurement
        log.info("Loading existing measurement data from disk.")
        obs_next = torch.load(data_file)
    # Visualize the observations to date
    data.visualize_obs(
        data_obs=obs_next,
        save_path=measure_dir / "observations.png",
    )

    # 8. Return new state
    new_state = ExperimentState(
        n_measured=n_measured + 1,
        data_obs=obs_next,
        posterior=state.posterior,
        nuisance_estimator=nuisance_estimator,
        theta_samples=theta_estimates,
        nuisance_samples=nuisance,
        x_grid=state.x_grid,
    )

    return new_state


def run_full_experiment(
    cfg: DictConfig,
    data,
    prior,
    nuisance_prior,
    acquisition_fn: Callable,
) -> ExperimentState:
    """Run the full experiment from initialization to completion."""
    # Make the initial measurement(s)
    data_obs = make_initial_measurement(
        data, data.x_grid, cfg.measurement.initial_measure, cfg.seed
    )

    # Sample from prior to initialize state
    theta_samples = prior.sample((cfg.sbi.n_samples,)).cpu()
    # Sample from nuisance prior
    nuisance_samples = nuisance_prior.sample(
        theta_samples,
        data_obs.unsqueeze(0).repeat(len(theta_samples), 1, 1),
        num_samples=1,
    ).cpu()

    state = ExperimentState(
        n_measured=cfg.measurement.initial_measure,
        data_obs=data_obs,
        posterior=prior,  # Initial posterior is just the prior
        nuisance_estimator=nuisance_prior,
        theta_samples=theta_samples,
        nuisance_samples=nuisance_samples,
        x_grid=data.x_grid,
    )

    # Main loop
    for n_measured in range(
        cfg.measurement.initial_measure, cfg.measurement.max_measure + 1
    ):
        measure_dir = Path(f"measurement_{n_measured}")
        measure_dir.mkdir(parents=True, exist_ok=True)

        # Run a single round
        log.info(f"Starting measurement round {n_measured}")
        state = run_experiment_round(
            state, cfg, data, prior, acquisition_fn, measure_dir
        )

    return state


def get_sbi_params(cfg: DictConfig, data) -> dict:
    """Extract SBI parameters from config."""
    return {
        "sbi_bs": cfg.sbi.batch_size,
        "sbi_lr": cfg.sbi.lr,
        "max_epochs": cfg.sbi.max_epochs,
        "stop_after_epochs": cfg.sbi.stop_after_epochs,
        "validation_fraction": cfg.sbi.validation_fraction,
        "prior_bounds": list(zip(data.prior_low, data.prior_high)),
        "simulator": data.simulator,
        "true_angles": data.true_thetas,
        "norm_theta": cfg.sbi.norm_theta,
        "model": cfg.sbi.model,
        "from_scratch": cfg.sbi.from_scratch,
    }


def one_step_experiment(
    cfg: DictConfig,
    data,
    prior,
    nuisance_prior,
    acquisition_fn: Callable,
) -> torch.Tensor:
    # This shouldn't be used anywhere except for shape checking/inference
    data_obs = make_initial_measurement(
        data, data.x_grid, cfg.measurement.initial_measure, cfg.seed
    )

    # Define a measurement directory
    measure_dir = Path("one_step_measurement")
    measure_dir.mkdir(parents=True, exist_ok=True)

    # Sample thetas from the prior
    theta_samples = prior.sample((cfg.sbi.n_samples,)).cpu()

    # 1. Run acquisition
    x_next = acquisition_fn(
        data_obs=data_obs,
        data_posterior=prior,
        nuisance_estimator=nuisance_prior,
        n_measured=1,  # Not really but needed for interface
        thetas=theta_samples,
        measure_dir=measure_dir,
        data=data,  # Pass data to acquisition function
    )

    return x_next
