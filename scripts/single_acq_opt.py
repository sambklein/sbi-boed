# Make some initial measurements and sample from the exact posterior with MCMC
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
from sbi_bax.experiments.base import DummyNuisanceEstimator
import logging
import time
import json


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="source_finding")
def main(cfg: DictConfig):
    # Set the random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    # Save the resolved configuration to a file
    with open("full_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    # Get the parent path where posterior inference was run
    load_dir = Path(cfg.output_dir)
    # Instantiate everything via Hydra
    log.info("Starting experiment.")
    # Instantiate
    data = hydra.utils.instantiate(cfg.data)
    # Load the true thetas just in case
    true_thetas = torch.load(load_dir / "true_thetas.pt", weights_only=True)
    data.true_thetas = true_thetas
    # Instantiate the acquisition function
    acquisition_fn = hydra.utils.instantiate(cfg.acquisition)
    # Load initial observations from disk
    initial_obs = torch.load(load_dir / "init_obs.pt", weights_only=True)
    # Get the number of points measured
    n_measured = initial_obs.shape[0]

    # Load theta estimates from disk
    theta_samples = torch.load(load_dir / "theta_estimates.pt", weights_only=True)

    # Load the posterior from disk
    posterior = torch.load(load_dir / "npe_posterior/posterior.pt", weights_only=False)

    # Update the x_grid if needed
    data.x_grid = data.make_x_grid(
        posterior=posterior, n_points=data.x_grid.shape[0], thetas=theta_samples
    )

    # Define the nuisance prior
    nuisance_prior = (
        data.nuisance_prior
        if hasattr(data, "nuisance_prior")
        else DummyNuisanceEstimator()
    )

    # Run acquisition (data saved internally)
    start_time = time.perf_counter()
    acquisition_fn(
        data_obs=initial_obs,
        data_posterior=posterior,  # This is only needed for NPE stacked flows
        nuisance_estimator=nuisance_prior,
        n_measured=n_measured,
        thetas=theta_samples,
        measure_dir=Path("."),
        data=data,  # Pass data to acquisition function
    )
    elapsed_time = time.perf_counter() - start_time

    timing_data = {
        "acquisition_optimization_seconds": elapsed_time,
        "acquisition_optimization_minutes": elapsed_time / 60,
    }
    with open("timing.json", "w") as f:
        json.dump(timing_data, f, indent=2)

    # On experiment completion, write a summary file (for snakemake)
    with open("single_acq_opt.txt", "w") as f:
        f.write(f"Experiment {cfg.exp_name} completed successfully.\n")
    log.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
