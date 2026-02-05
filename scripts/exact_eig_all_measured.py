# Code to check performance of first measurement designs for SPCE
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

import torch

from sbi_bax.experiments.nle_sequential import NleExactAcquisition
import matplotlib.pyplot as plt

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
    # Define some uncofigured parameters
    n_mc = 5_00
    n_designs = 1_000
    # Instantiate the Nle exact acquisition class
    nle_exact = NleExactAcquisition(n_mc=n_mc, batch_size=1024)
    # Define the design grid
    x_grid = data.x_grid.cpu()
    # Sample designs from the grid
    xi = x_grid[:n_designs].cpu()
    # Get the default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the observation history
    data_obs = torch.load("data_obs.pt").cpu()
    # Get the number of observations
    n_obs = data_obs.shape[0]
    # Create workdir
    workdir = Path("exact_eig")
    workdir.mkdir(exist_ok=True)
    # Create a store for the best designs and EIGs
    store = {"designs": [], "eig": [], "actual_designs": [], "eig_observed": []}
    # Going to loop through each design and compute EIG
    for i in range(1, n_obs):
        # Load the posterior theta samples
        theta_samples = torch.load(f"measurement_{i}/sbi_dt/samples_post.pt").cpu()
        # Get the number of posterior samples
        n_post_samples = theta_samples.shape[0]
        assert n_post_samples >= n_mc, "Not enough posterior samples for MC estimation"
        # Extract the actual design taken at this measurement
        actual_design = data_obs[i, 1]
        # Add this to xi so we can get the EIG at the actual design
        xi_r = torch.cat([xi, actual_design.view(1, 1)], dim=0)
        xi = xi_r.unique(dim=0)
        # Set up the nle compute loss function
        nle_exact._setup(data_obs, None, data, n_obs, theta_samples)
        # Compute EIG for all designs at this measurement
        loss, _ = nle_exact._compute_loss(xi.to(device), n_mc)
        eig = -loss  # since loss is negative EIG
        # Store the best design and EIG observed
        best_design = xi[torch.argmax(eig)]
        eig_observed = eig[(xi == actual_design).flatten()].squeeze()
        store["designs"].append(best_design.cpu())
        store["eig"].append(torch.max(eig).cpu())
        store["actual_designs"].append(actual_design.cpu())
        store["eig_observed"].append(eig_observed.cpu())
        # Plot designs vs EIG
        plot_path = workdir / f"eig_measurement_{i}.png"
        plt.figure()
        # Plot the EIG
        plt.plot(
            xi.cpu().numpy(), eig.cpu().numpy(), marker="o", linestyle="", label="EIG"
        )
        # Plot the best design
        plt.axvline(
            best_design.cpu().item(), color="r", linestyle="--", label="Best Design"
        )
        # Plot the actual design taken
        plt.axvline(
            actual_design.cpu().item(),
            color="g",
            linestyle=":",
            label="Design Selected",
        )
        plt.legend()
        plt.title(f"EIG vs Design at Measurement {i}")
        plt.xlabel("Design")
        plt.ylabel("EIG")
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
    # Save the store as a torch file
    print(store["designs"])
    torch.save(store, workdir / "eig_all_measurements.pt")
    log.info("Experiment completed.")


if __name__ == "__main__":
    # This ensures types are respected in main
    main()
