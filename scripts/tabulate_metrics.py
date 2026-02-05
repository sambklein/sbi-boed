import argparse
import pickle
import pandas as pd
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
    parser = argparse.ArgumentParser(
        description="Summarize aggregated experiment results"
    )
    parser.add_argument(
        "input_file",
        help="Path to aggregated_experiments.pkl file",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save the summary CSV (default: results)",
    )
    parser.add_argument(
        "--output-tag",
        default="summary.csv",
        help="Filename for the summary CSV (default: summary.csv)",
    )
    args = parser.parse_args()

    # Load aggregated data
    with open(args.input_file, "rb") as f:
        aggregated_data = pickle.load(f)

    # Compute statistics grouped by method and dimension
    summary_rows = []
    for method, experiments in aggregated_data.items():
        # Group by dimension
        by_dim = {}
        by_n_design = {}
        for exp in experiments:
            dim = exp.get(
                "dimension", "none"
            )  # Use "none" for experiments without dimensions
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(exp)
            cfg = exp["full_config"]
            n_design = cfg["acquisition"]["n_designs"]
            if n_design not in by_n_design:
                by_n_design[n_design] = []
            by_n_design[n_design].append(exp)

        # Create row for each (method, dimension, n_designs) combination
        for dim, dim_exps in by_dim.items():
            for n_design, n_design_exps in by_n_design.items():
                # Filter experiments that have both this dimension and n_designs
                combined_exps = [exp for exp in dim_exps if exp in n_design_exps]

                spce_low_values = [
                    exp["spce_low"][-1] for exp in combined_exps if "spce_low" in exp
                ]
                avg_spce_low = (
                    np.mean(spce_low_values) if spce_low_values else float("nan")
                )
                median_spce_low = (
                    np.median(spce_low_values) if spce_low_values else float("nan")
                )
                std_spce_low = (
                    np.std(spce_low_values)
                    if len(spce_low_values) > 1
                    else float("nan")
                )
                # Get the standard error of the mean if more than one experiment
                sem_spce_low = (
                    np.std(spce_low_values) / np.sqrt(len(spce_low_values))
                    if len(spce_low_values) > 1
                    else float("nan")
                )

                # Extract timing values
                timing_values = [
                    exp["timing"]
                    for exp in combined_exps
                    if "timing" in exp and exp["timing"] is not None
                ]
                avg_timing = np.mean(timing_values) if timing_values else float("nan")
                sem_timing = (
                    np.std(timing_values) / np.sqrt(len(timing_values))
                    if len(timing_values) > 1
                    else float("nan")
                )

                summary_rows.append(
                    {
                        "acquisition_method": method,
                        "dimension": dim,
                        "n_designs": n_design,
                        "avg_spce_low": avg_spce_low,
                        "median_spce_low": median_spce_low,
                        "std_spce_low": std_spce_low,
                        "n_experiments": len(combined_exps),
                        "sem_spce_low": sem_spce_low,
                        "avg_timing_minutes": avg_timing,
                        "sem_timing_minutes": sem_timing,
                    }
                )
                # Plot a histogram of spce_low values for this method/dim/n_designs
                if spce_low_values:
                    plt.figure()
                    plt.hist(spce_low_values, bins=10, alpha=0.7)
                    plt.title(
                        f"SPCE Low Histogram\nMethod: {method}, Dim: {dim}, n_designs: {n_design}"
                    )
                    plt.xlabel("SPCE Low")
                    plt.ylabel("Frequency")
                    hist_dir = Path(args.output_dir) / "aa_histograms"
                    hist_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(
                        hist_dir
                        / f"{args.output_tag}_hist_spce_low_{method}_dim_{dim}_ndesigns_{n_design}.png",
                        dpi=150,
                    )
                    plt.close()

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    # Sort by dimension (putting "none" last) then by method
    summary_df = summary_df.sort_values(
        by=["dimension", "acquisition_method"],
        key=lambda x: x.map(
            lambda v: (1e10, v)
            if v == "none"
            else (v, v)
            if isinstance(v, (int, float))
            else (0, v)
        ),
    )

    print(summary_df)
    summary_df.to_csv(f"{args.output_dir}/{args.output_tag}.csv", index=False)

    print(f"\nSaved summary table to {args.output_dir}/{args.output_tag}.csv")

    # Check if there is more than one n_designs for plotting
    plt_dir = Path(args.output_dir) / "plots"
    plt_dir.mkdir(parents=True, exist_ok=True)
    if summary_df["n_designs"].nunique() > 1:
        # For each dimension plot avg_spce_low vs n_designs for each method with error bars
        # (This part is optional and can be implemented as needed)

        # Colorblind-friendly color palette (Okabe-Ito colors)
        colors = [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ]
        # Alternative markers for better distinction
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        # Set publication-quality defaults
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 11,
                # "axes.titlesize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
                # "figure.figsize": (4.5, 3),  # Changed from (6, 4)
                "figure.figsize": (4, 3),  # Changed from (6, 4)
                # "lines.linewidth": 1.2,
                # "lines.markersize": 5,
                # "errorbar.capsize": 2,
            }
        )

        for dim in summary_df["dimension"].unique():
            # SPCE plot
            fig, ax = plt.subplots()
            dim_df = summary_df[summary_df["dimension"] == dim]

            methods = sorted(dim_df["acquisition_method"].unique())
            # Filter out any methods that start with "nle_contrastive"
            methods = [m for m in methods if not m.startswith("nle_contrastive")]
            for idx, method in enumerate(methods):
                method_df = dim_df[dim_df["acquisition_method"] == method].sort_values(
                    "n_designs"
                )
                ax.errorbar(
                    method_df["n_designs"],
                    method_df["avg_spce_low"],
                    yerr=method_df["sem_spce_low"],
                    label=transform_label(method),
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    capsize=3,
                    alpha=0.9,
                )

            ax.set_xscale("log")
            ax.set_xlabel("Number of Designs")
            ax.set_ylabel("Average SPCE")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            ax.set_ylim(0.5, 1.0)  # Set y-axis limits for better visibility
            plt.savefig(plt_dir / f"spce_low_dim_{dim}.pdf", bbox_inches="tight")
            plt.close(fig)

            # Timing plot
            fig, ax = plt.subplots()
            dim_df = summary_df[summary_df["dimension"] == dim]

            for idx, method in enumerate(methods):
                method_df = dim_df[dim_df["acquisition_method"] == method].sort_values(
                    "n_designs"
                )
                ax.errorbar(
                    method_df["n_designs"],
                    method_df["avg_timing_minutes"],
                    yerr=method_df["sem_timing_minutes"],
                    label=transform_label(method),
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    capsize=3,
                    alpha=0.9,
                )

            ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Number of Designs")
            ax.set_ylabel("Runtime (minutes)")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(plt_dir / f"timing_dim_{dim}.png", dpi=300, bbox_inches="tight")
            plt.savefig(plt_dir / f"timing_dim_{dim}.pdf", bbox_inches="tight")
            plt.close(fig)

        # Plot SPCE values across all measurements
        # Organize data for plotting: {dimension: {method: {n_measured: [spce_values]}}}
        plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for method, experiments in aggregated_data.items():
            for exp in experiments:
                dim = exp.get("dimension", "none")

                # Get spce_low values across all measurements
                spce_values = exp.get("spce_low", [])

                # Each measurement index represents n_measured + 1 (starting from 1 initial measurement)
                for n_measured, spce_val in enumerate(spce_values, start=1):
                    plot_data[dim][method][n_measured].append(spce_val)

        # Create plots for each dimension
        for dim in plot_data.keys():
            fig, ax = plt.subplots(figsize=(10, 6))

            for method in plot_data[dim].keys():
                # Collect data points
                n_measured_list = []
                avg_spce_list = []
                sem_spce_list = []

                for n_measured in sorted(plot_data[dim][method].keys()):
                    spce_vals = plot_data[dim][method][n_measured]
                    n_measured_list.append(n_measured)
                    avg_spce_list.append(np.mean(spce_vals))
                    if len(spce_vals) > 1:
                        sem_spce_list.append(
                            np.std(spce_vals) / np.sqrt(len(spce_vals))
                        )
                    else:
                        sem_spce_list.append(0)

                # Plot with error bars
                ax.errorbar(
                    n_measured_list,
                    avg_spce_list,
                    yerr=sem_spce_list,
                    label=method,
                    marker="o",
                    capsize=5,
                )

            ax.set_title(f"SPCE Low vs Number of Measurements (Dimension: {dim})")
            ax.set_xlabel("Number of Measurements")
            ax.set_ylabel("Average SPCE Low")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"spce_vs_measurements_dim_{dim}_{args.output_tag}.png"
            plt.savefig(plt_dir / filename, dpi=150)
            plt.close(fig)
            print(f"Saved plot to {plt_dir / filename}")

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
            fig, ax = plt.subplots()

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
                    marker="o",
                    capsize=5,
                )

            ax.set_title(
                f"Median SPCE vs Measurements (Dim: {dim}, n_designs: {n_design})"
            )
            ax.set_xlabel("Number of Measurements")
            ax.set_ylabel("Median SPCE Low")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = (
                f"median_spce_dim_{dim}_ndesigns_{n_design}_{args.output_tag}.png"
            )
            plt.savefig(plt_dir / filename, dpi=150)
            plt.close(fig)
            print(f"Saved plot to {plt_dir / filename}")


if __name__ == "__main__":
    main()
