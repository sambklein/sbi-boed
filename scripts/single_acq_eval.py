# Code to check performance of first measurement designs for SPCE
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sbi_bax.models.s_pce import compute_eig


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="source_finding")
def main(cfg: DictConfig) -> None:
    # Initialize the experiment
    log.info("Starting experiment.")
    # Set the random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    # Get the parent path where posterior inference was run
    load_dir = Path(
        cfg.output_dir
    )  # Get the parent path where posterior inference was run
    # Instantiate everything as in initial experiment
    data = hydra.utils.instantiate(cfg.data)
    # Extract the true log_likelihood function
    log_likelihood = data.simulator.log_prob
    # Extract the prior
    prior = data.prior

    # Load thetas from disk for evaluation
    theta_samples = torch.load(load_dir / "theta_estimates.pt", weights_only=True)
    n_traces = 1_00
    batch_size = 1
    L_contrast = 100_000
    true_thetas = theta_samples[:n_traces].cpu()
    # Load the best design from disk
    xi = torch.tensor(np.loadtxt("best_x.txt"), dtype=torch.float32)
    # Repeat designs for each theta
    xi_rep = xi.unsqueeze(0).repeat(n_traces, 1).cpu()
    # For every theta, make a measurement at each design
    obs = data.simulator(xi_rep, true_thetas).cpu()
    # Load the posterior from disk
    prior = torch.load(load_dir / "npe_posterior/posterior.pt", weights_only=False)
    # Unsqueeze to add time dimension
    xi_rep = xi_rep.unsqueeze(1)
    # Unsqueeze data to give it a batch dimension
    obs = obs.unsqueeze(1).unsqueeze(1)
    # Run the evaluation in batches to save memory
    bounds = []
    for i in tqdm(range(0, len(true_thetas), batch_size)):
        batch_true_thetas = true_thetas[i : i + batch_size]
        batch_obs = obs[i : i + batch_size]
        batch_xi_rep = xi_rep[i : i + batch_size]
        info_bounds = compute_eig(
            prior,
            log_likelihood,
            batch_true_thetas,
            batch_xi_rep,
            batch_obs,
            L=L_contrast,
        ).squeeze()
        bounds.append(info_bounds)
    info_stacked = torch.tensor(bounds)
    # Save the information bounds to disk
    torch.save(info_stacked, "eig_eval.pt")

    # Plot a histogram of the EIGs
    plt.figure(figsize=(8, 6))
    plt.hist(info_stacked.numpy(), bins=20, alpha=0.7)
    plt.xlabel("EIG")
    plt.ylabel("Frequency")
    plt.title("Histogram of EIGs for First Measurement Designs")
    plt.savefig("eig_histogram.png")
    plt.close()

    # Estimate the sample size needed to get a good estimate of the EIG
    mean_eig = info_stacked.mean().item()
    std_eig = info_stacked.std().item()
    # 95% confidence interval
    se = std_eig / np.sqrt(n_traces)
    conf_interval = 1.96 * se  # 95% CI
    log.info(f"Estimated EIG: {mean_eig:.4f} ± {conf_interval:.4f} (95% CI)")

    # Estimate n_samples needed for desired standard error
    desired_se = 0.1 * mean_eig  # SE should be 10% of mean
    n_samples_needed = (std_eig / desired_se) ** 2
    log.info(f"Estimated samples needed for SE = 10% of mean: {int(n_samples_needed)}")
    # Estimate n_samples needed for desired CI width
    desired_ci_half_width = 0.1 * mean_eig  # 95% CI half-width
    desired_se = desired_ci_half_width / 1.96
    n_samples_needed = (std_eig / desired_se) ** 2
    log.info(f"Estimated samples needed for desired CI width: {int(n_samples_needed)}")


if __name__ == "__main__":
    # This ensures types are respected in main
    main()
