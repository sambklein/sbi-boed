import yaml
from pathlib import Path
import os
from sbi_bax.utils.snakemake import get_sweep_experiments, build_param_overrides
configfile: f"{workflow.basedir}/config/paper_many_experiments.yaml"

# Extract experiment configurations
base_configs = config["defaults"]["base_configs"]
results_dir = config.get("results_dir", "sequential")
acquisitions = config["defaults"]["acquisition"]

# -----------------------------------------------------
# Build experiment sweep definitions
# -----------------------------------------------------
sweep_experiments = get_sweep_experiments(config, base_configs, acquisitions)
experiment_names = [exp["name"] for exp in sweep_experiments]
exp_lookup = {exp["name"]: exp for exp in sweep_experiments}

# -----------------------------------------------------
# Global Targets
# -----------------------------------------------------
rule all:
    input:
        # Experiments
        expand(f"results/{results_dir}/{{experiment}}/full_config.yaml", experiment=experiment_names),
        expand(f"results/{results_dir}/{{experiment}}/eval_info.json", experiment=experiment_names),

        # Tables
        expand(f"results/{results_dir}/table_{{base_config}}.csv", base_config=base_configs),


# -----------------------------------------------------
# Sequential (per-acquisition) experiment runs
# -----------------------------------------------------
rule run_experiment:
    output:
        config=f"results/{results_dir}/{{experiment}}/full_config.yaml",
        data=f"results/{results_dir}/{{experiment}}/data_obs.pt",
        summary=f"results/{results_dir}/{{experiment}}/experiment_summary.txt"
    params:
        config_dir=f"{workflow.basedir}/config",
        script=f"{workflow.basedir}/scripts/run_functional.py",
        base_config=lambda wc: exp_lookup[wc.experiment]["base_config"],
        seed=lambda wc: exp_lookup[wc.experiment]["seed"],
        acquisition=lambda wc: exp_lookup[wc.experiment]["acquisition"],
        dimension=lambda wc: exp_lookup[wc.experiment]["dimension"],
        fixed_overrides=lambda wc: build_param_overrides(
            config,
            exp_lookup[wc.experiment]["base_config"],
            exp_lookup[wc.experiment]["dimension"],
            exp_lookup[wc.experiment]["n_designs"]
        ),
        workdir=f"{os.getcwd()}/results/{results_dir}"
    log:
        f"logs/{results_dir}/{{experiment}}.log"
    group:
        "compute"
    shell:
        """
        pixi run -e cuda python {params.script} \
            --config-path {params.config_dir} \
            --config-name {params.base_config} \
            exp_name={wildcards.experiment} \
            seed={params.seed} \
            acquisition={params.acquisition} \
            output_dir={params.workdir} \
            {params.fixed_overrides}
        """

# -----------------------------------------------------
# Evaluation rule
# -----------------------------------------------------
rule run_evaluation:
    input:
        script=f"{workflow.basedir}/scripts/eval_sequential.py",
        config=f"results/{results_dir}/{{experiment}}/full_config.yaml",
        summary=f"results/{results_dir}/{{experiment}}/experiment_summary.txt"
    output:
        data=f"results/{results_dir}/{{experiment}}/eval_info.json"
    params:
        config_dir=f"{workflow.basedir}/config",
        base_config=lambda wc: exp_lookup[wc.experiment]["base_config"],
        seed=lambda wc: exp_lookup[wc.experiment]["seed"],
        acquisition=lambda wc: exp_lookup[wc.experiment]["acquisition"],
        dimension=lambda wc: exp_lookup[wc.experiment]["dimension"],
        fixed_overrides=lambda wc: build_param_overrides(
            config,
            exp_lookup[wc.experiment]["base_config"],
            exp_lookup[wc.experiment]["dimension"],
            exp_lookup[wc.experiment]["n_designs"]
        ),
        workdir=f"{os.getcwd()}/results/{results_dir}"
    log:
        f"logs/{results_dir}/{{experiment}}_eval.log"
    group:
        "compute"
        # lambda wc: f"{exp_lookup[wc.experiment]['base_config']}"
    shell:
        """
        pixi run -e cuda python {input.script} \
            --config-path {params.config_dir} \
            --config-name {params.base_config} \
            exp_name={wildcards.experiment} \
            seed={params.seed} \
            acquisition={params.acquisition} \
            output_dir={params.workdir} \
            {params.fixed_overrides}
        """

# -----------------------------------------------------
# Aggregation and Table Rules
# -----------------------------------------------------
rule aggregate_by_base_config:
    input:
        configs=lambda wc: [
            f"results/{results_dir}/{exp['name']}/full_config.yaml"
            for exp in sweep_experiments
            if exp["base_config"] == wc.base_config
        ],
        data=lambda wc: [
            f"results/{results_dir}/{exp['name']}/eval_info.json"
            for exp in sweep_experiments
            if exp["base_config"] == wc.base_config
        ],
        script=f"{workflow.basedir}/scripts/aggregate_experiments.py",
    output:
        f"results/{results_dir}/{{base_config}}/aggregated_experiments.pkl"
    params:
        results_dir=results_dir
    group:
        "aggregation"
    shell:
        """
        pixi run -e cuda python {input.script} \
        {input.configs} \
        --output-dir results/{params.results_dir}/{wildcards.base_config}
        """

rule make_table:
    input:
        agg_results=f"results/{results_dir}/{{base_config}}/aggregated_experiments.pkl",
        script=f"{workflow.basedir}/scripts/tabulate_metrics.py"
    output:
        f"results/{results_dir}/table_{{base_config}}.csv"
    params:
        results_dir=results_dir,
        output_tag="table_{base_config}"
    group:
        "aggregation"
    shell:
        """
        pixi run -e cuda python {input.script} \
        {input.agg_results} \
        --output-dir results/{params.results_dir} \
        --output-tag {params.output_tag}
        """
        # --output-file results/{params.results_dir}/table_{wildcards.base_config}.csv

rule clean:
    shell:
        "rm -rf results/ logs/"
