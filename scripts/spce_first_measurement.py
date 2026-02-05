# Code to check performance of first measurement designs for SPCE
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

    # # Load the original thetas from disk (also in experiment class but this is safer)
    # true_thetas = torch.load("true_thetas.pt").cpu()
    # # Load the observation history
    # data_obs = torch.load("data_obs.pt").cpu()
    # # Infer the number of observations
    # n_obs = data_obs.shape[0]

    # Sample thetas from prior for evaluation
    # n_eval_thetas = 1_000
    n_eval_thetas = 1_000
    max_designs = 10
    batch_size = 64
    true_thetas = prior.sample((n_eval_thetas,)).cpu()
    # Grab designs from x_grid
    # xi = data.x_grid.unsqueeze(0).repeat(n_eval_thetas, 1, 1).cpu()
    # xi = data.x_grid[:max_designs]
    xi = torch.linspace(
        data.data_low[0],
        data.data_high[0],
        steps=max_designs,
    ).unsqueeze(1)
    # For every theta, make a measurement at each design
    obs = torch.cat(
        [
            data.simulator(xi, true_thetas[i].unsqueeze(0)).cpu()
            for i in range(n_eval_thetas)
        ],
        dim=0,
    )
    # Repeat designs for each theta
    xi_rep = xi.repeat(n_eval_thetas, 1).cpu()
    # Unsqueeze to add time dimension
    xi_rep = xi_rep.unsqueeze(1)
    # Repeat thetas for each design
    true_thetas = true_thetas.repeat_interleave(max_designs, dim=0).cpu()
    # Unsqueeze data to give it a batch dimension
    obs = obs.unsqueeze(1).unsqueeze(1)
    # Run the evaluation in batches to save memory
    bounds = []
    for i in range(0, len(true_thetas), batch_size):
        batch_true_thetas = true_thetas[i : i + batch_size]
        batch_obs = obs[i : i + batch_size]
        batch_xi_rep = xi_rep[i : i + batch_size]
        info_bounds = compute_eig(
            prior,
            log_likelihood,
            batch_true_thetas,
            batch_xi_rep,
            batch_obs,
            L=int(cfg.eval.n_contrast),
        ).squeeze()
        bounds.append(info_bounds)
    info_stacked = torch.cat(bounds, dim=0)
    n_now = info_stacked.shape[0]

    # Plot the design vs EIG
    plt.figure(figsize=(8, 6))
    plt.plot(
        xi_rep.squeeze().numpy()[:n_now],
        info_stacked.numpy(),
        marker="o",
        linestyle="",
        color="blue",
        label="EIG",
    )
    plt.xlabel("Design")
    plt.ylabel("Expected Information Gain (nats)")
    plt.title("EIG vs Design for First Measurement")
    plt.legend()
    plt.grid(True)
    plt.savefig("eig_vs_design.png")
    plt.close()

    # Average across thetas for each design
    avg_bounds = info_stacked[:n_now].view(-1, max_designs).mean(dim=0)
    # Plot the average EIG vs design
    plt.figure(figsize=(8, 6))
    plt.plot(
        xi.squeeze().numpy(),
        avg_bounds.numpy(),
        marker="o",
        linestyle="-",
        color="red",
        label="Average EIG",
    )
    plt.xlabel("Design")
    plt.ylabel("Average Expected Information Gain (nats)")
    plt.title("Average EIG vs Design for First Measurement")
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_eig_vs_design.png")
    plt.close()

    log.info("Evaluation complete.")


if __name__ == "__main__":
    # This ensures types are respected in main
    main()
