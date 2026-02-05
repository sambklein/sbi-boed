# A script to load posterior samples and plot them along with the result of running optimization to find the best design.
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


class HandlerGradientCircle(HandlerPatch):
    def __init__(self, cmap="viridis", num_segments=200, **kw):
        self.cmap = plt.get_cmap(cmap)
        self.num_segments = num_segments
        super().__init__(**kw)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Calculate circle center and radius
        center_x = width / 2 - xdescent
        center_y = height / 2 - ydescent
        radius = min(width, height) / 2

        # Create horizontal strips for vertical gradient
        colors = self.cmap(np.linspace(1.2, -0.2, self.num_segments))

        # Create clip path circle
        circle_path = mpatches.Circle((center_x, center_y), radius, transform=trans)

        patches = []
        for i in range(self.num_segments):
            # Calculate y positions for this strip
            y_frac_top = i / self.num_segments
            y_frac_bottom = (i + 1) / self.num_segments

            # Convert to y coordinates relative to circle center
            y_top = radius * (1 - 2 * y_frac_top)
            y_bottom = radius * (1 - 2 * y_frac_bottom)

            # Only draw the part of the rectangle that's inside the circle
            rect = mpatches.Rectangle(
                (center_x - radius, center_y + y_bottom),
                2 * radius,
                y_top - y_bottom,
                facecolor=colors[i],
                edgecolor="none",
                transform=trans,
                clip_path=circle_path,
                alpha=1.0,
            )
            patches.append(rect)

        return patches


def visualize_obs(
    data_obs,
    true_thetas,
    n_source,
    physical_dim,
    optimized_designs,
    final_eig,
    save_path,
):
    """
    Visualize the observed data with detector locations, true sources, and proposed designs with EIG values.

    Args:
        data_obs: Observed data tensor
        true_thetas: True theta values tensor
        n_source: Number of sources
        physical_dim: Physical dimension (2 or 3)
        optimized_designs: Proposed detector locations with EIG scores
        final_eig: EIG values for each proposed design
        save_path: Path to save the figure
    """
    thetas = true_thetas.view(n_source, physical_dim)
    # Extract measurement locations and intensities from data_obs
    total_intensity, xi = data_obs[..., 0], data_obs[..., 1:]
    xi = xi.squeeze(0) if xi.dim() > 1 else xi
    total_intensity = total_intensity.exp()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Plot proposed designs colored by EIG
    scatter = ax.scatter(
        optimized_designs[:, 0],
        optimized_designs[:, 1],
        c=final_eig - final_eig.min(),
        s=np.clip(final_eig * 50, 50, 100),
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
        # label="Candidate Designs",
    )

    # Plot true sources
    source_markers = ["x", "+"]
    for i in range(n_source):
        ax.scatter(
            thetas[i, 0],
            thetas[i, 1],
            c="red",
            marker=source_markers[i],
            s=100,
            linewidths=2,
            label=f"Source {i + 1}",
            zorder=10,
        )

    # Plot measured locations
    ax.scatter(
        xi[:, 0],
        xi[:, 1],
        c="orange",
        marker="*",
        s=80,
        linewidths=2,
        label=r"Observed $\xi$",
        zorder=10,
    )

    # # Add colorbar for EIG
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('EIG', rotation=270, labelpad=15)

    # # Labels
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    # Set axis limits and ensure square aspect ratio
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal", adjustable="box")

    # Add inset colorbar that doesn't change figure size
    cax = ax.inset_axes([0.75, 0.05, 0.05, 0.25])
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label("EIG", rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Create custom legend entry for candidate designs
    # Create a dummy patch for the gradient circle legend entry
    gradient_circle = mpatches.Circle((0, 0), 1)

    legend_elements = [
        gradient_circle,  # Replace the Line2D with gradient circle
        ax.get_legend_handles_labels()[0][2],  # Observed Design
        ax.get_legend_handles_labels()[0][0],  # Sources 1
        ax.get_legend_handles_labels()[0][1],  # Source 2
    ]

    # Update the legend call to include handler_map:
    ax.legend(
        handles=legend_elements,
        labels=[r"Candidate $\xi$", r"Observed $\xi$", "Source 1", "Source 2"],
        handler_map={gradient_circle: HandlerGradientCircle()},
        loc="upper center",
        ncol=2,
        frameon=False,
        facecolor="none",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_posterior_2d_histogram(
    samples,
    true_location,
    obs_locations,
    save_path,
    xlabel="X",
    ylabel="Y",
    bins=50,
    figsize=(3.5, 3.5),  # Standard single column width for papers
    cmap="Blues",
    density=True,
    axis_limits=(-3, 3),
    source_label="Source",
    source_marker="x",
    cbar_label="Density",
):
    """
    Plot 2D histogram of posterior samples with true location marked.

    Args:
        samples: Tensor of shape (n_samples, 2) containing x, y coordinates
        true_location: Tensor of shape (2,) containing true x, y coordinates
        save_path: Path to save the figure
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of bins for the histogram
        figsize: Figure size in inches (width, height)
        cmap: Colormap for the histogram
        density: If True, normalize the histogram to create a probability density
        axis_limits: Tuple of (min, max) for both axes
        source_label: Label for the true location marker
        source_marker: Marker style for the true location
        cbar_label: Label for the colorbar
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy for plotting
    samples_np = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    true_loc_np = (
        true_location.cpu().numpy()
        if isinstance(true_location, torch.Tensor)
        else true_location
    )

    # Create 2D histogram
    h = ax.hist2d(
        samples_np[:, 0],
        samples_np[:, 1],
        bins=bins,
        cmap=cmap,
        density=density,
        edgecolors="none",
        range=[axis_limits, axis_limits],
    )

    # Add inset colorbar that doesn't change figure size
    cax = ax.inset_axes([0.75, 0.05, 0.05, 0.25])
    cbar = plt.colorbar(h[3], cax=cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Mark true location
    ax.scatter(
        true_loc_np[0],
        true_loc_np[1],
        c="red",
        marker=source_marker,
        s=100,
        linewidths=2,
        label=source_label,
        zorder=10,
    )
    # Mark observation locations
    ax.scatter(
        obs_locations[:, 0],
        obs_locations[:, 1],
        c="orange",
        marker="*",
        s=80,
        linewidths=2,
        label=r"Observed $\xi$",
        zorder=10,
    )
    # Add legend in upper right
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.5,
        edgecolor="none",  # No border, just background
        facecolor="white",
    )

    # # Labels and title
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)

    # Set axis limits and ensure square aspect ratio
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect("equal", adjustable="box")

    # Make sure the layout is tight
    plt.tight_layout()

    # Save with high DPI for paper quality
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_posterior_sources(
    data_obs, theta_samples, true_thetas, n_source, physical_dim, plot_dir
):
    """
    Plot posterior distributions for each source separately.

    Args:
        theta_samples: Posterior samples of shape (n_samples, n_source * physical_dim)
        true_thetas: True theta values of shape (n_source * physical_dim,)
        n_source: Number of sources
        physical_dim: Physical dimension (should be 2 for 2D plots)
        plot_dir: Directory to save plots
    """
    # Reshape samples and true values
    samples_reshaped = theta_samples.view(-1, n_source, physical_dim)
    true_reshaped = true_thetas.view(n_source, physical_dim)
    # Separate out design variables
    _, xi = data_obs[..., 0], data_obs[..., 1:]

    # Plot each source
    for i in range(n_source):
        source_samples = samples_reshaped[:, i, :]  # (n_samples, 2)
        true_location = true_reshaped[i, :]  # (2,)

        plot_posterior_2d_histogram(
            samples=source_samples,
            true_location=true_location,
            obs_locations=xi,
            save_path=plot_dir / f"posterior_source_{i + 1}.pdf",  # Use PDF for papers
            xlabel="X",
            ylabel="Y",
            bins=50,
            figsize=(3.5, 3.5),  # Single column width
            cmap="Blues",
            density=True,
            source_label=f"Source {i + 1}",
            # source_label=f"$\\theta_{{{i + 1}}}$",
            # cbar_label=f"$p(\\theta_{{{i + 1}}})$",
            source_marker=["x", "+"][i],
        )

        print(f"Saved posterior plot for source {i + 1}")


def main():
    # Define the directory under which we will work
    workdir = Path("results/n_designs_short/source_finding_smart_dim2_ndes128_seed87")
    # Define directory to save plots
    plot_dir = Path("results/paper_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load the posterior samples
    theta_samples = torch.load(workdir / "theta_estimates.pt")
    # Load the true theta values
    true_thetas = torch.load(workdir / "true_thetas.pt")
    # Load the observations
    data_obs = torch.load(workdir / "init_obs.pt")
    # Load the optimized designs
    # This job was launched using a launch.json entry. Look there for how to reproduce.
    design_info = torch.load(workdir / "contrastive_forplot/warmup/final_designs.pt")
    optimized_designs = design_info["designs"]
    final_eig = design_info["final_eig"]

    # Plot posterior distributions for each source
    plot_posterior_sources(
        data_obs=data_obs,
        theta_samples=theta_samples,
        true_thetas=true_thetas,
        n_source=2,
        physical_dim=2,
        plot_dir=plot_dir,
    )

    # Visualize the observations with proposed designs
    visualize_obs(
        data_obs=data_obs,
        true_thetas=true_thetas,
        n_source=2,
        physical_dim=2,
        optimized_designs=optimized_designs,
        final_eig=final_eig,
        save_path=plot_dir / "observations_with_designs.pdf",
    )


if __name__ == "__main__":
    main()
