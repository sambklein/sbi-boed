import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_nested_value(data, key_path):
    """Get value from nested dict using dot notation like 'measurement.acquisition_method'"""
    keys = key_path.split(".")
    value = data
    for key in keys:
        value = value[key]
    return value


def name_to_label(name):
    """Convert class names to readable labels"""
    return {
        "sbi_bax.experiments.random_select.RandomAcquisition": "Random",
        "sbi_bax.experiments.bo.UcbSbi": "SBI UCB",
        "sbi_bax.experiments.bax.InfoBax": "InfoBAX",
        "sbi_bax.experiments.bax.MeanBax": "MeanBAX",
        "sbi_bax.experiments.bo.StandardBoUcb": "GP UCB",
    }.get(name, name)


def create_cumulative_plot(grouped_data, output_dir, plot_type="maximum"):
    """Create cumulative maximum or minimum plot"""
    if plot_type == "maximum":
        accumulate_func = np.maximum.accumulate
        ylabel = "Maximum Measured Value"
        title = "Cumulative Maximum"
        filename = "cumulative_maximum.pdf"
    elif plot_type == "minimum":
        accumulate_func = np.minimum.accumulate
        ylabel = "Minimum Measured Value"
        title = "Cumulative Minimum"
        filename = "cumulative_minimum.pdf"
    else:
        raise ValueError("plot_type must be 'maximum' or 'minimum'")

    plt.figure(figsize=(10, 6))
    for key, experiments in grouped_data.items():
        if not experiments:  # Skip empty groups
            continue

        y_measured_arrays = [exp["y_measured"] for exp in experiments]
        min_length = min(len(y) for y in y_measured_arrays)

        cumulative_values = []
        for y_measured in y_measured_arrays:
            y_truncated = y_measured[:min_length].reshape(-1)
            cumulative = accumulate_func(y_truncated)
            cumulative_values.append(cumulative)

        cumulative_values = np.array(cumulative_values)
        avg_cumulative = np.mean(cumulative_values, axis=0)
        std_cumulative = np.std(cumulative_values, axis=0)

        measurement_numbers = np.arange(1, len(avg_cumulative) + 1)
        plt.plot(
            measurement_numbers,
            avg_cumulative,
            label=name_to_label(str(key)),
            marker="o",
        )
        plt.fill_between(
            measurement_numbers,
            avg_cumulative - std_cumulative,
            avg_cumulative + std_cumulative,
            alpha=0.2,
        )

    plt.xlabel("Number of Measurements")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    if len(avg_cumulative) > 0:
        plt.xticks(range(1, len(avg_cumulative) + 1))
    plt.tight_layout()

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    log.info(f"Saved {title.lower()} plot to {output_file}")
    plt.close()


def create_plots(grouped_data, output_dir):
    """Create all measurement history plots"""

    # Plot 1: Average measured values
    plt.figure(figsize=(10, 6))
    for key, experiments in grouped_data.items():
        if not experiments:  # Skip empty groups
            continue

        min_length = min(len(exp["y_measured"]) for exp in experiments)
        y_measured = np.array(
            [exp["y_measured"][:min_length].reshape(-1) for exp in experiments]
        )

        avg_y = np.mean(y_measured, axis=0)
        std_y = np.std(y_measured, axis=0)

        measurement_numbers = np.arange(1, len(avg_y) + 1)
        plt.plot(measurement_numbers, avg_y, label=name_to_label(str(key)), marker="o")
        plt.fill_between(measurement_numbers, avg_y - std_y, avg_y + std_y, alpha=0.2)

    plt.xlabel("Measurement Index")
    plt.ylabel("Measured Value")
    plt.title("Average Measured Values")
    plt.legend(frameon=False)
    if len(avg_y) > 0:
        plt.xticks(range(1, len(avg_y) + 1))
    plt.tight_layout()

    output_file = output_dir / "average_measured_values.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    log.info(f"Saved average measured values plot to {output_file}")
    plt.close()

    # Plot 2: Cumulative maximum
    create_cumulative_plot(grouped_data, output_dir, "maximum")

    # Plot 3: Cumulative minimum
    create_cumulative_plot(grouped_data, output_dir, "minimum")


def main():
    parser = argparse.ArgumentParser(
        description="Plot measurement history from aggregated experiments"
    )
    parser.add_argument(
        "pickle_file", help="Path to the aggregated experiments pickle file"
    )
    parser.add_argument(
        "--output-dir", default="results/aggregated", help="Output directory for plots"
    )
    parser.add_argument(
        "--group-by",
        default="experiment._target_",
        help="Config key to group experiments by (e.g., experiment._target_)",
    )

    args = parser.parse_args()

    # Load aggregated data
    log.info(f"Loading aggregated data from {args.pickle_file}")
    with open(args.pickle_file, "rb") as f:
        aggregated_data = pickle.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by the specified key
    grouped_data = defaultdict(list)
    for exp_data in aggregated_data:
        key = get_nested_value(exp_data["full_config"], args.group_by)
        grouped_data[key].append(exp_data)

    log.info(f"Found {len(grouped_data)} different groups: {list(grouped_data.keys())}")

    # Create plots
    create_plots(grouped_data, output_dir)

    log.info("All plots created successfully")


if __name__ == "__main__":
    main()
