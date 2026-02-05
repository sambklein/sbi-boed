def get_sweep_experiments(config, base_configs, acquisitions=None) -> list[dict]:
    experiments = []
    seeds = config["defaults"]["seeds"]
    if "n_jobs" in config["defaults"]:
        n_jobs = config["defaults"]["n_jobs"]
        seeds = list(range(42, 42 + n_jobs))
    if acq_not_passed := acquisitions is None:
        acquisitions = [""]
    for base_config in base_configs:
        overrides = config.get("config_overrides", {}).get(base_config, {})

        # Get sweep parameters (default to [None] if not specified)
        dims = overrides.get("dimensions", [None])
        n_designs_list = overrides.get("n_designs", [None])

        # Normalize to list if single value
        if not isinstance(n_designs_list, list):
            n_designs_list = [n_designs_list]

        for seed in seeds:
            for acq in acquisitions:
                for dim in dims:
                    for n_designs in n_designs_list:
                        acq_name = acq.split(".")[-1]

                        # Build experiment name with optional suffixes
                        name_parts = [base_config]
                        if not acq_not_passed:
                            name_parts.append(acq_name)

                        if dim is not None:
                            name_parts.append(f"dim{dim}")
                        if n_designs is not None and len(n_designs_list) > 1:
                            name_parts.append(f"ndes{n_designs}")
                        name_parts.append(f"seed{seed}")
                        exp_name = "_".join(name_parts)

                        experiments.append(
                            {
                                "name": exp_name,
                                "base_config": base_config,
                                "seed": seed,
                                "acquisition": acq if not acq_not_passed else None,
                                "dimension": dim,
                                "n_designs": n_designs,
                            }
                        )
    return experiments


def build_param_overrides(config, base_config, dimension=None, n_designs=None):
    params = config["fixed_params"].copy()

    if "config_overrides" in config and base_config in config["config_overrides"]:
        overrides = config["config_overrides"][base_config].copy()
        # Remove sweep keys since they're not hydra overrides
        overrides.pop("dimensions", None)
        overrides.pop("n_designs", None)
        params.update(overrides)

    # Add sweep overrides if specified
    if dimension is not None:
        params["data.simulator.physical_dim"] = dimension
    if n_designs is not None:
        params["acquisition.n_designs"] = n_designs

    overrides = []
    for key, value in params.items():
        if value is None or isinstance(value, list):
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        overrides.append(f"{key}={value}")

    return " \\\n            ".join(overrides)
