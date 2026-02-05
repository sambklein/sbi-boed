import matplotlib.pyplot as plt
import numpy as np


def plot_eig_and_components(
    x_vals,
    eig_x,
    term_one,
    term_two,
    y_true,
    data_obs,
    x_next,
    y_next,
    fig_name,
):
    """
    Plot the Expected Information Gain (EIG) and its components as a function of x.

    Args:
        x_vals (torch.Tensor): The x values.
        eig_x (list): The EIG values for each x.
        term_one (list): The first term of the EIG (H(y_x | D_t)).
        term_two (list): The second term of the EIG (E[H(y_x | θ, D_t)]).
        y_true (numpy.ndarray): The true function values.
        data_obs (torch.Tensor): The observed data.
        x_next (torch.Tensor): The next x value to measure.
        y_next (torch.Tensor): The next y value to measure.
        n_measure (int): The current measurement number.
        sub_dir (Path): The directory to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Expected Information Gain on the left y-axis
    ax1.plot(
        x_vals.cpu().numpy(), eig_x, label="Expected Information Gain", color="blue"
    )
    ax1.plot(
        x_vals.cpu().numpy(),
        term_one,
        label=r"$H(y_x | D_t)$",
        linestyle="--",
        color="orange",
    )
    ax1.plot(
        x_vals.cpu().numpy(),
        term_two,
        label=r"$E_{p(e_A|D_t)}[H(y_x | D_t, e_A)]$",
        linestyle="--",
        color="green",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("Expected Information Gain", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a second axis for the true function
    ax2 = ax1.twinx()
    ax2.plot(x_vals.cpu().numpy(), y_true, color="black", label="True Function")
    ax2.scatter(
        data_obs[:, 1].cpu().numpy(),
        data_obs[:, 0].cpu().numpy(),
        color="red",
        label="Observed Data",
    )
    ax2.scatter(
        x_next.cpu().numpy(),
        y_next.cpu().numpy(),
        color="green",
        label="Next Measurement",
    )
    ax2.set_ylabel("True Function Value", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combine legends from both subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        ncol=4,
        frameon=False,
    )

    # Add title and save the figure
    plt.savefig(fig_name)
    plt.close()


def plot_progress(history, best_eig, logs, figname):
    """
    Plot optimization progress with main loss and arbitrary number of model logs.

    Args:
        history: List of mean loss values per step
        best_eig: List of best EIG values per step
        logs: Dict of {log_name: [values]} for model training logs
        figname: Path to save figure
    """
    # Calculate grid dimensions
    n_plots = 1 + len(logs)  # Main loss plot + all model logs
    n_cols = min(3, n_plots)  # Max 3 columns
    n_rows = int(np.ceil(n_plots / n_cols))

    # Create figure with grid layout
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,  # Always return 2D array
    )

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Plot main loss (first subplot)
    axes_flat[0].plot(history, label="Mean loss")
    axes_flat[0].plot(best_eig, label="Best EIG")
    axes_flat[0].legend()
    axes_flat[0].set_xlabel("Step")
    axes_flat[0].set_ylabel("Loss")
    axes_flat[0].set_title("Optimization Progress")

    # Plot each model log
    for i, (name, values) in enumerate(logs.items(), start=1):
        axes_flat[i].plot(values, label=name)
        axes_flat[i].legend()
        axes_flat[i].set_xlabel("Batch")
        axes_flat[i].set_ylabel("Log prob")
        axes_flat[i].set_title(name)

    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
    plt.close(fig)
