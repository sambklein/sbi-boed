import argparse
import pickle
import torch
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_dimension_from_config(cfg: OmegaConf) -> int | None:
    """Extract dimension from config. Returns None if no dimension specified."""
    try:
        if (
            hasattr(cfg, "data")
            and hasattr(cfg.data, "simulator")
            and hasattr(cfg.data.simulator, "physical_dim")
        ):
            return cfg.data.simulator.physical_dim
        elif hasattr(cfg, "data") and hasattr(cfg.data, "design_dim"):
            return cfg.data.design_dim
        elif hasattr(cfg, "dimension"):
            return cfg.dimension
        elif hasattr(cfg, "design_dim"):
            return cfg.design_dim
        else:
            return None
    except AttributeError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results for tabulate_metrics.py"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Experiment directories from ablations.smk results (e.g., results/performance_short/pharmacokinetic_seed42)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/aggregated",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--output-file",
        default="aggregated_experiments.pkl",
        help="Output pickle filename",
    )
    parser.add_argument(
        "--filter-dimension",
        type=int,
        default=None,
        help="Only aggregate experiments with this dimension (optional)",
    )
    args = parser.parse_args()

    aggregated_data = defaultdict(list)
    skipped_count = 0
    processed_count = 0

    for exp_dir in args.experiment_dirs:
        exp_path = Path(exp_dir)

        # Load base config
        config_path = exp_path / "full_config.yaml"
        if not config_path.exists():
            log.warning(f"No config found at {config_path}, skipping")
            skipped_count += 1
            continue

        try:
            exp_cfg = OmegaConf.load(config_path)
            mode = "sequential"
            exp_dimension = get_dimension_from_config(exp_cfg)

            # Check dimension filter
            if args.filter_dimension is not None:
                if exp_dimension != args.filter_dimension:
                    log.info(
                        f"Skipping {exp_path}: dimension {exp_dimension} != {args.filter_dimension}"
                    )
                    skipped_count += 1
                    continue

            # Process each acquisition subdirectory
            for acq_dir in exp_path.iterdir():
                if not acq_dir.is_dir():
                    continue

                eig_file = acq_dir / "eig_eval.pt"
                timing_file = acq_dir / "acquisition_timing.txt"

                if not eig_file.exists():
                    log.debug(f"No eig_eval.pt found in {acq_dir}, skipping")
                    continue

                # Load EIG evaluations
                eig_values = torch.load(eig_file, weights_only=True)

                # Extract timing if available
                timing_seconds = None
                if timing_file.exists():
                    with open(timing_file, "r") as f:
                        line = f.readline()
                        # Parse "Acquisition optimization time: X.XX seconds"
                        if "seconds" in line:
                            timing_seconds = float(
                                line.split(":")[1].strip().split()[0]
                            )

                # Get acquisition method name from directory name
                acquisition_method = acq_dir.name
                # Get if a distance penalty was used
                exp_cfg = OmegaConf.load(acq_dir / "full_config.yaml")
                distance_penalty = getattr(
                    exp_cfg.acquisition, "distance_penalty", None
                )
                if distance_penalty is not None and distance_penalty:
                    acquisition_method += "_with_distance_penalty"

                # Get if scale_and_shift was used
                scale_and_shift = getattr(exp_cfg.acquisition, "scale_and_shift", True)
                if scale_and_shift:
                    acquisition_method += "_with_scale_and_shift"
                else:
                    acquisition_method += "_no_scale_and_shift"

                # Extract timing if available
                timing_file = acq_dir / "timing.json"
                timing = None
                if timing_file.exists():
                    with open(timing_file, "r") as f:
                        timing = json.load(f)["acquisition_optimization_minutes"]

                # Create experiment data matching aggregate_experiments.py format
                exp_data = {
                    "mode": mode,
                    "dimension": exp_dimension,
                    "full_config": OmegaConf.to_container(exp_cfg, resolve=True),
                    "spce_low": [
                        eig_values.mean().item()
                    ],  # Single measurement, so list of 1
                    "eig_mean": eig_values.mean().item(),
                    "eig_std": eig_values.std().item(),
                    "eig_median": eig_values.median().item(),
                    "n_traces": len(eig_values),
                    "timing": timing,
                }

                if timing_seconds is not None:
                    exp_data["acquisition_time_seconds"] = timing_seconds

                # Group by acquisition method
                aggregated_data[acquisition_method].append(exp_data)
                processed_count += 1

                log.info(
                    f"Processed {exp_path.name}/{acquisition_method} (mode={mode}, dim={exp_dimension})"
                )

        except Exception as e:
            log.error(f"Failed to process {exp_path}: {e}")
            skipped_count += 1
            continue

    # Save aggregated data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    with open(output_file, "wb") as f:
        pickle.dump(aggregated_data, f)

    log.info(f"Saved aggregated data to {output_file}")
    log.info(
        f"Aggregation completed successfully. Processed {len(aggregated_data)} acquisition methods "
        f"({processed_count} experiments total, {skipped_count} skipped)."
    )
    if args.filter_dimension is not None:
        log.info(f"Filtered by dimension: {args.filter_dimension}")


if __name__ == "__main__":
    main()
