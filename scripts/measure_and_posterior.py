# Make some initial measurements and sample from the exact posterior with MCMC
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from sbi_bax.experiments.npe_sequential import NpeAcquisition
from sbi_bax.experiments.functional import get_sbi_params
from sbi_bax.models.transformer import TransformerEncoder
from sbi_bax.utils.sbi import run_sbi
from sbi_bax.models.flow import DeconditionedFlow
from pathlib import Path

import logging


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(
    version_base=None, config_path="../config", config_name="source_finding_cinit"
)
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
    # Save the true thetas to disk for evaluation purposes
    torch.save(data.true_thetas, "true_thetas.pt")
    # Call the data to make initial measurements
    initial_obs = data.initial_measurement(cfg.seed)
    # Visualize the observations
    data.visualize_obs(
        data_obs=initial_obs,
        save_path=Path("initial_observations.png"),
    )
    # Get the number of points measured
    n_measured = initial_obs.shape[0]
    # Save the initial observations
    torch.save(initial_obs, "init_obs.pt")
    # workdir = Path("mcmc")
    # workdir.mkdir(exist_ok=True)
    # NOTE: alternative way to define posteriors, uses MCMC and exact likelihood
    # # Sample from the exact posterior using MCMC
    # _, theta_estimates = build_nle_mcmc_posterior(
    #     ExactLikelihood(data),
    #     data.prior,
    #     data,
    #     n_measured,
    #     initial_obs,
    #     workdir,
    #     {"from_scratch": cfg.measurement.from_scratch},
    #     cfg.sbi.n_samples,
    # )
    # torch.save(theta_estimates, "theta_estimates_mcmc.pt")

    # Train an NPE posterior, this is used for the stacked flows set up in acquisition
    npe_acq = hydra.utils.instantiate(cfg.acquisition)
    assert isinstance(npe_acq, NpeAcquisition), (
        "Acquisition function must be NPE for this script."
    )
    # Build training dataset
    theta_samples = data.prior.sample((cfg.sbi.n_samples,)).cpu()
    with torch.no_grad():
        # Get training data from data object
        train_obs, train_xi = data.build_train_set(
            theta_samples,
            initial_obs,
            n_measured,
        )
    # Setup posterior (inline from _setup_posterior)
    sbi_params = get_sbi_params(cfg, data)
    measure_dir = Path("npe_posterior")
    measure_dir.mkdir(exist_ok=True)
    # Split out no_ea data
    data_no_ea = data.combine_no_ea(train_obs, train_xi, n_measured).squeeze(1)
    # Build the embedding net
    embedding_net = TransformerEncoder(
        input_dim=data_no_ea.shape[-1],
        trial_net_output_dim=128,
    )
    posterior, theta_estimates = run_sbi(
        data_no_ea,
        theta_samples,
        data.prior,
        embedding_net,
        data_obs=initial_obs,
        measure_dir=measure_dir,
        initial_model=None,
        n_samples=npe_acq.n_thetas,
        hyperspherical=data.hyperspherical,
        **sbi_params,
    )
    # Wrap in DeconditionedFlow for NPE
    posterior = DeconditionedFlow(
        posterior.posterior_estimator.net.cpu(),
        initial_obs.cpu().unsqueeze(0),
    )
    # Save the full posterior to disk
    torch.save(posterior, "npe_posterior/posterior.pt")
    # Save the theta estimates to disk
    torch.save(theta_estimates, "theta_estimates.pt")
    log.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
