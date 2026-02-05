import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_expected_information_gain(
    angle_grid: torch.Tensor,
    eig_x: torch.Tensor,
    true_angles: torch.Tensor,
    measure_dir: Path,
):
    """Plot the expected information gain (EIG) for each angle."""
    # NOTE: these figures are not actually very helpful, should think of something else.
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    eig_x = np.array(eig_x)

    # Define consistent color scale
    vmin, vmax = eig_x.min(), eig_x.max()

    for i in range(3):
        for j in range(3):
            if i == j:
                ax[i, j].scatter(angle_grid[:, i].cpu().numpy(), eig_x, label="EIG")
                # Plot the true angle as a vertical line
                ax[i, j].axvline(
                    true_angles[i].cpu().numpy(),
                    color="red",
                    linestyle="--",
                    label="True Angle",
                )
                ax[i, j].legend(frameon=False)
                ax[i, j].set_title(f"EIG for {['x', 'y', 'z'][i]} angle")
                ax[i, j].set_xlabel("Angle (radians)")
                ax[i, j].set_ylabel("Expected Information Gain")
            elif i < j:
                ax[i, j].scatter(
                    angle_grid[:, i].cpu().numpy(),
                    angle_grid[:, j].cpu().numpy(),
                    c=eig_x,
                    cmap="viridis",
                    s=10,
                    vmin=vmin,  # Set consistent min
                    vmax=vmax,  # Set consistent max
                )
                ax[i, j].set_xlabel(f"Angle {['x', 'y', 'z'][i]}")
                ax[i, j].set_ylabel(f"Angle {['x', 'y', 'z'][j]}")
            else:
                ax[i, j].axis("off")

    # Add a colorbar with consistent scale
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax),  # Use explicit normalization
            cmap="viridis",
        ),
        ax=ax,
        orientation="horizontal",
    )
    cbar.set_label("Expected Information Gain")
    # Set the title for the entire figure
    plt.suptitle("Expected Information Gain for Each Angle")
    # Save the figure
    fig.savefig(measure_dir / "expected_information_gain.png", bbox_inches="tight")
