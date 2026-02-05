import argparse
from collections import defaultdict
import json
import pickle
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_dimension_from_config(cfg: OmegaConf) -> int | None:
    """
    Extract dimension from config. Returns None if no dimension specified.
    """
    try:
        # Try common dimension keys
        if hasattr(cfg, "data") and hasattr(cfg.data, "simulator.physical_dim"):
            return cfg.data.simulator.physical_dim
        elif hasattr(cfg, "data") and hasattr(cfg.data, "design_dim"):
            return cfg.data.design_dim
        elif hasattr(cfg, "dimension"):
            return cfg.dimension
        elif hasattr(cfg, "design_dim"):
            return cfg.design_dim
        else:
            return cfg.data.simulator.physical_dim
    except AttributeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "config_files",
        nargs="+",
        help="Config files (full_config.yaml) from experiment directories",
    )
    parser.add_argument(
        "--output-dir",
        default="results/aggregated",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--output-file",
        default="aggregated_experiments.pkl",
        help="Output filename",
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

    for config_file in args.config_files:
        config_path = Path(config_file)
        exp_dir = config_path.parent
        mode = "sequential"

        try:
            log.info(f"Loading experiment from {exp_dir} (mode={mode})")

            exp_cfg = OmegaConf.load(config_path)

            # Check dimension filter
            exp_dimension = get_dimension_from_config(exp_cfg)
            if args.filter_dimension is not None:
                if exp_dimension != args.filter_dimension:
                    log.info(
                        f"Skipping {exp_dir}: dimension {exp_dimension} != {args.filter_dimension}"
                    )
                    skipped_count += 1
                    continue

            # data_obs = torch.load(exp_dir / "data_obs.pt")

            with open(exp_dir / "eval_info.json") as f:
                info_dict = json.load(f)

            exp_data = {
                "mode": mode,
                "dimension": exp_dimension,
                "full_config": OmegaConf.to_container(exp_cfg, resolve=True),
                # "data_obs": data_obs.cpu().numpy(),
                **info_dict,
            }

            acquisition_method = exp_cfg.acquisition._target_.split(".")[-1]
            aggregated_data[acquisition_method].append(exp_data)

            log.info(
                f"Aggregated experiment: {exp_cfg.exp_name} (mode={mode}, dim={exp_dimension})"
            )

        except Exception as e:
            log.error(f"Failed to process {exp_dir}: {e}")
            continue

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    with open(output_file, "wb") as f:
        pickle.dump(aggregated_data, f)

    log.info(f"Saved aggregated data to {output_file}")
    log.info(
        f"Aggregation completed successfully. Processed {len(aggregated_data)} acquisition methods "
        f"({sum(len(v) for v in aggregated_data.values())} experiments total, {skipped_count} skipped)."
    )
    if args.filter_dimension is not None:
        log.info(f"Filtered by dimension: {args.filter_dimension}")


if __name__ == "__main__":
    main()
