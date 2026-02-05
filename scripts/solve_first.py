import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from sbi_bax.experiments.base import DummyNuisanceEstimator
from sbi_bax.experiments.functional import one_step_experiment

import logging


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="ces")
def main(cfg: DictConfig):
    # Set the random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    # Save the resolved configuration to a file
    with open("full_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    # Instantiate everything via Hydra
    log.info("Starting experiment.")
    # Instantiate
    data = hydra.utils.instantiate(cfg.data)
    # Write the true thetas to disk for evaluation purposes
    torch.save(data.true_thetas, "true_thetas.pt")
    acquisition_fn = hydra.utils.instantiate(cfg.acquisition)

    # Find the optimal first measurement to make
    nuisance_prior = (
        data.nuisance_prior
        if hasattr(data, "nuisance_prior")
        else DummyNuisanceEstimator()
    )
    x_first = one_step_experiment(cfg, data, data.prior, nuisance_prior, acquisition_fn)

    # Save the final observation in the top level directory
    torch.save(x_first, "x_first.pt")

    # On experiment completion, write a summary file (for snakemake)
    with open("experiment_summary.txt", "w") as f:
        f.write(f"Experiment {cfg.exp_name} completed successfully.\n")
    log.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
