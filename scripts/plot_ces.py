import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def transform_label(method):
    parts = method.split("_")
    tag = parts[0]
    if len(parts) > 1 and parts[1] == "contrastive":
        tag += "_contrastive"
    return {
        "npe": "NPE",
        "nle": "NLE",
        "nle_contrastive": "NLE-C",
        "contrastive": "NRE",
        "NpeContrastiveAcquisition": "NPE-NRE",
        "random": "Random",
    }.get(tag, method)


def main():
    # Load comparison results from RL BOED
    # rl_boed_path = Path(
    #     "/sdf/group/mli/samklein/code/RL-BOED/logs/boed_results/ces/evaluation_20000.pt"
    # )
    # rl_boed_data = torch.load(rl_boed_path, map_location="cpu", weights_only=True)
    # NOTE: rerunning RL-BOED code resulted in a worse performance than reported in their paper,
    # so we are using the original reported numbers here.
    # # Average along the first axis
    # rl_boed_avg = rl_boed_data.mean(dim=1).numpy()
    # # Get the standard error along the first axis
    # rl_boed_sem = rl_boed_data.std(dim=1).numpy() / np.sqrt(rl_boed_data.shape[1])
    # CES RL-BOED mean, taken from their paper
    rl_boed_avg = np.array([4.1, 7.9, 10.9, 12.5, 13.1, 13.3, 13.5, 13.7, 13.8, 13.97])
    rl_boed_sem = np.array([0.02, 0.05, 0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01, 0.06])
    # Do the same for DAD
    dad_boed_mean = np.array([4.1, 6.8, 7.8, 9.5, 10.01, 10.5, 10.7, 10.7, 10.9, 10.9])
    dad_boed_sem = np.array(
        [0.02, 0.05, 0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01, 0.06]
    )
    # Random mean, written out here because why not
    random_mean = np.array([2.15, 3.9, 5.12, 5.83, 6.35, 6.9, 7.5, 7.9, 8.01, 8.2])
    random_sem = np.array([0.1, 0.15, 0.12, 0.18, 0.2, 0.25, 0.22, 0.3, 0.28, 0.3])

    # Load aggregated data
    with open("results/ces_performance/ces/aggregated_experiments.pkl", "rb") as f:
        aggregated_data = pickle.load(f)

    # Check if there is more than one n_designs for plotting
    plt_dir = Path("results/paper_plots") / "plots"
    plt_dir.mkdir(parents=True, exist_ok=True)

    # Plot median SPCE values across all measurements grouped by dimension and n_designs
    # Organize data for plotting: {dimension: {n_designs: {method: {n_measured: [spce_values]}}}}
    plot_data_median = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for method, experiments in aggregated_data.items():
        for exp in experiments:
            dim = exp.get("dimension", "none")
            cfg = exp["full_config"]
            n_design = cfg["acquisition"]["n_designs"]

            # Get spce_low values across all measurements
            spce_values = exp.get("spce_low", [])

            # Each measurement index represents n_measured + 1 (starting from 1 initial measurement)
            for n_measured, spce_val in enumerate(spce_values, start=1):
                plot_data_median[dim][n_design][method][n_measured].append(spce_val)

    # Create plots for each dimension and n_designs combination
    for dim in plot_data_median.keys():
        for n_design in plot_data_median[dim].keys():
            fig, ax = plt.subplots(figsize=(4, 3))

            # Add the CES RL-BOED baseline
            ax.errorbar(
                np.arange(1, len(rl_boed_avg) + 1),
                rl_boed_avg,
                yerr=rl_boed_sem,
                label="RL-BOED",
                marker="s",
                capsize=5,
                color="#E69F00",
            )

            # Add the Random baseline
            ax.errorbar(
                np.arange(1, len(random_mean) + 1),
                random_mean,
                yerr=random_sem,
                label="Random",
                marker="o",
                capsize=5,
                color="gray",
            )

            # Plot DAD BOED baseline
            ax.errorbar(
                np.arange(1, len(dad_boed_mean) + 1),
                dad_boed_mean,
                yerr=dad_boed_sem,
                label="DAD",
                marker=">",
                capsize=5,
                color="#56B4E9",
            )

            for method in plot_data_median[dim][n_design].keys():
                # Collect data points
                n_measured_list = []
                median_spce_list = []
                sem_spce_list = []

                for n_measured in sorted(
                    plot_data_median[dim][n_design][method].keys()
                ):
                    spce_vals = plot_data_median[dim][n_design][method][n_measured]
                    n_measured_list.append(n_measured)
                    median_spce_list.append(np.median(spce_vals))
                    if len(spce_vals) > 1:
                        sem_spce_list.append(
                            np.std(spce_vals) / np.sqrt(len(spce_vals))
                        )
                    else:
                        sem_spce_list.append(0)

                # Plot with error bars
                ax.errorbar(
                    n_measured_list,
                    median_spce_list,
                    yerr=sem_spce_list,
                    label=transform_label(method),
                    marker="^",
                    color="#009E73",
                    capsize=5,
                )
            ax.set_xlabel("Number of Measurements")
            ax.set_ylabel("Average sPCE")
            ax.legend(frameon=False, ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = "ces.pdf"
            plt.savefig(plt_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot to {plt_dir / filename}")


if __name__ == "__main__":
    main()
