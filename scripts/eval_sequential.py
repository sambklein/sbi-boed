# General script to evaluate runs
import json
import hydra
from omegaconf import DictConfig
import logging

import torch

from sbi_bax.models.s_pce import compute_eig


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="source_finding")
def main(cfg: DictConfig) -> None:
    # Initialize the experiment
    log.info("Starting experiment.")
    # Instantiate everything as in initial experiment
    data = hydra.utils.instantiate(cfg.data)
    # Extract the true log_likelihood function
    log_likelihood = data.simulator.log_prob
    # Extract the prior
    prior = data.prior
    # Load the original thetas from disk (also in experiment class but this is safer)
    true_thetas = torch.load("true_thetas.pt").cpu()
    # Load the observation history
    data_obs = torch.load("data_obs.pt").cpu()
    # Infer the number of observations
    n_obs = data_obs.shape[0]
    # Split the observations into designs and outputs
    obs, xi = data.split_obs(data_obs, n_obs)
    # Unsqueeze data to give it a batch dimension
    true_thetas = true_thetas.unsqueeze(0)
    obs = obs.unsqueeze(0)
    # Run the evaluation
    info_bounds = compute_eig(
        prior, log_likelihood, true_thetas, xi, obs, L=int(cfg.eval.n_contrast)
    ).squeeze()
    # Convert each of the bounds to scalars
    info_dict = {
        "spce_low": info_bounds.cpu().numpy().tolist(),
    }

    # # Load theta samples from the posterior from disk
    # posterior_thetas = torch.load(
    #     f"measurement_{cfg.measurement.max_measure}/sbi_dt/samples_post.pt"
    # ).cpu()
    # # Compute posterior metrics
    # # Median distance to true theta
    # median_distance = torch.median(
    #     torch.norm(posterior_thetas - true_thetas.unsqueeze(1), dim=-1)
    # ).item()
    # # 90% credible interval volume (axis-aligned box)
    # lower_bound = torch.quantile(posterior_thetas, 0.05, dim=0)
    # upper_bound = torch.quantile(posterior_thetas, 0.95, dim=0)
    # credible_interval_volume = torch.prod(upper_bound - lower_bound).item()
    # # Log the metrics
    # log.info(f"Median distance to true theta: {median_distance:.4f}")
    # log.info(f"90% credible interval volume: {credible_interval_volume:.4f}")

    # # Save the eval info to disk
    # info_dict["median_distance"] = median_distance
    # info_dict["credible_interval_volume"] = credible_interval_volume
    with open("eval_info.json", "w") as f:
        json.dump(info_dict, f, indent=2)
    print(info_dict)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    # This ensures types are respected in main
    main()
