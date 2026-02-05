# General script to evaluate runs
import json
import hydra
from omegaconf import DictConfig
import logging

import torch
import matplotlib.pyplot as plt
from sbi_bax.models.s_pce import compute_eig


# Set up logging
log = logging.getLogger(__name__)
# Suppress ArviZ preview logs
logging.getLogger("arviz.preview").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="pharmacokinetic")
def main(cfg: DictConfig) -> None:
    # Initialize the experiment
    log.info("Starting experiment.")
    # Instantiate everything as in initial experiment
    data = hydra.utils.instantiate(cfg.data)
    # Extract the true log_likelihood function
    log_likelihood = data.simulator.log_prob
    # Extract the prior
    prior = data.prior
    # Define the static designs to evaluate
    xi = torch.tensor([17.56, 0.3889, 5.3970, 23.9907, 0.3223])
    n_test = 100
    info_bounds = []
    for _ in range(n_test):
        # Sample thetas from prior for evaluation
        true_thetas = prior.sample((1,)).cpu()
        # Make an observation at each design
        obs = data.simulator(xi.unsqueeze(1), true_thetas.repeat(xi.shape[0], 1)).cpu()
        obs = obs.unsqueeze(0)
        # Run the evaluation
        info_bounds.append(
            compute_eig(
                prior,
                log_likelihood,
                true_thetas,
                xi.unsqueeze(0),
                obs,
                L=int(cfg.eval.n_contrast),
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # Stack info bounds
    info_bounds_mat = torch.tensor(info_bounds)
    # Save the matrix of info bounds
    torch.save(info_bounds_mat, "eval_static_info_bounds.pt")
    # For each design, plot a histogram of the info bounds
    _, axs = plt.subplots(1, xi.shape[0], figsize=(4 * xi.shape[0], 4))
    for i in range(xi.shape[0]):
        axs[i].hist(info_bounds_mat[:, i].numpy(), bins=20, alpha=0.7)
        axs[i].set_title(f"Design {xi[i].item():.2f}")
        axs[i].set_xlabel("EIG")
        axs[i].set_ylabel("Frequency")
    plt.tight_layout()
    plot_path = "eval_info_histograms.png"
    plt.savefig(plot_path)
    plt.close()

    # Take the mean over tests
    info_bounds = torch.mean(info_bounds_mat, dim=0).numpy().tolist()
    # Convert each of the bounds to scalars
    info_dict = {
        "spce_low": info_bounds,
        "spce_std": torch.std(info_bounds_mat, dim=0).numpy().tolist(),
        "spce_std_err": (
            torch.std(info_bounds_mat, dim=0)
            / torch.sqrt(torch.tensor(info_bounds_mat.shape[0], dtype=torch.float32))
        )
        .numpy()
        .tolist(),
    }

    with open("eval_static.json", "w") as f:
        json.dump(info_dict, f, indent=2)
    print(info_dict)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    # This ensures types are respected in main
    main()
