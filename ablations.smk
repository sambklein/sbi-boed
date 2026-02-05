import yaml
from pathlib import Path
import os
from sbi_bax.utils.snakemake import get_sweep_experiments, build_param_overrides
configfile: f"{workflow.basedir}/config/paper_ablations.yaml"

# Extract experiment configurations
base_configs = config["defaults"]["base_configs"]
results_dir = f'{config.get("results_dir", "sequential")}_short'
acquisitions = config["defaults"]["acquisition"]
dist_penaltys = config["defaults"].get("dist_penalty", [True])

# -----------------------------------------------------
# Build experiment sweep definitions
# -----------------------------------------------------
sweep_experiments = get_sweep_experiments(config, base_configs)
experiment_names = [exp["name"] for exp in sweep_experiments]
exp_lookup = {exp["name"]: exp for exp in sweep_experiments}

# -----------------------------------------------------
# Global Targets
# -----------------------------------------------------
rule all:
    input:
        # Experiments
        expand(f"results/{results_dir}/{{experiment}}/theta_estimates.pt", experiment=experiment_names),
        expand(f"results/{results_dir}/{{experiment}}/{{acquisition}}_{{dist_penalty}}/single_acq_opt.txt", experiment=experiment_names, acquisition=acquisitions, dist_penalty=dist_penaltys),
        expand(f"results/{results_dir}/{{experiment}}/{{acquisition}}_{{dist_penalty}}/eig_eval.pt", experiment=experiment_names, acquisition=acquisitions, dist_penalty=dist_penaltys),
        # Per-config aggregation and tables
        expand(f"results/{results_dir}/{{base_config}}/ablations_summary.csv", base_config=base_configs),
        # # Overall aggregation and tables
        # f"results/{results_dir}/ablations_summary.csv",

rule make_posterior:
    output:
        theta_samples=f"results/{results_dir}/{{experiment}}/theta_estimates.pt",
    params:
        script=f"{workflow.basedir}/scripts/measure_and_posterior.py",
        config_dir=f"{workflow.basedir}/config",
        base_config=lambda wc: exp_lookup[wc.experiment]["base_config"],
        seed=lambda wc: exp_lookup[wc.experiment]["seed"],
        acquisition="npe",  # Fixes the posterior method to NPE.
        dimension=lambda wc: exp_lookup[wc.experiment]["dimension"],
        fixed_overrides=lambda wc: build_param_overrides(
            config,
            exp_lookup[wc.experiment]["base_config"],
            exp_lookup[wc.experiment]["dimension"],
            exp_lookup[wc.experiment]["n_designs"]
        ),
        workdir=f"{os.getcwd()}/results/{results_dir}"
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

rule acquisition_optimization:
    input:
        rules.make_posterior.output.theta_samples
    output:
        completed=f"results/{results_dir}/{{experiment}}/{{acquisition}}_{{dist_penalty}}/single_acq_opt.txt",
    params:
        script=f"{workflow.basedir}/scripts/single_acq_opt.py",
        config_dir=f"{workflow.basedir}/config",
        base_config=lambda wc: exp_lookup[wc.experiment]["base_config"],
        seed=lambda wc: exp_lookup[wc.experiment]["seed"],
        acquisition=lambda wc: wc.acquisition,
        dimension=lambda wc: exp_lookup[wc.experiment]["dimension"],
        dist_penalty=lambda wc: wc.dist_penalty,
        fixed_overrides=lambda wc: build_param_overrides(
            config,
            exp_lookup[wc.experiment]["base_config"],
            exp_lookup[wc.experiment]["dimension"],
            exp_lookup[wc.experiment]["n_designs"],
        ),
        workdir=lambda wc: f"{os.getcwd()}/results/{results_dir}/{wc.experiment}"
    group:
        "eval_{experiment}_{acquisition}"
    shell:
        """
        pixi run -e cuda python {params.script} \
            --config-path {params.config_dir} \
            --config-name {params.base_config} \
            exp_name={wildcards.acquisition}_{wildcards.dist_penalty} \
            seed={params.seed} \
            acquisition={params.acquisition} \
            output_dir={params.workdir} \
            acquisition.dist_penalty={params.dist_penalty} \
            {params.fixed_overrides}
        """

rule run_evaluation:
    input:
        rules.acquisition_optimization.output.completed,
        script=f"{workflow.basedir}/scripts/single_acq_eval.py",
    output:
        completed=f"results/{results_dir}/{{experiment}}/{{acquisition}}_{{dist_penalty}}/eig_eval.pt",
    params:
        config_dir=f"{workflow.basedir}/config",
        base_config=lambda wc: exp_lookup[wc.experiment]["base_config"],
        workdir=lambda wc: f"{os.getcwd()}/results/{results_dir}/{wc.experiment}"
    group:
        "eval_{experiment}_{acquisition}"
    shell:
        """
        pixi run -e cuda python {input.script} \
            --config-path {params.config_dir} \
            --config-name {params.base_config} \
            exp_name={wildcards.acquisition}_{wildcards.dist_penalty} \
            output_dir={params.workdir}
        """
    
rule aggregate_ablations_by_config:
    input:
        eigs=lambda wc: expand(
            f"results/{results_dir}/{{experiment}}/{{acquisition}}_{{dist_penalty}}/eig_eval.pt",
            experiment=[exp["name"] for exp in sweep_experiments if exp["base_config"] == wc.base_config],
            acquisition=acquisitions,
            dist_penalty=dist_penaltys
        ),
    output:
        f"results/{results_dir}/{{base_config}}/aggregated_experiments.pkl"
    params:
        script=f"{workflow.basedir}/scripts/aggregate_ablations.py",
        results_dir=results_dir,
        experiment_dirs=lambda wc: [
            f"results/{results_dir}/{exp['name']}/" 
            for exp in sweep_experiments 
            if exp["base_config"] == wc.base_config
        ],
    group:
        "evaluate"
    shell:
        """
        python {params.script} {params.experiment_dirs} \
            --output-dir results/{params.results_dir}/{wildcards.base_config}/ \
            --output-file aggregated_experiments.pkl
        """

rule tabulate_ablations_by_config:
    input:
        agg=f"results/{results_dir}/{{base_config}}/aggregated_experiments.pkl",
        script=f"{workflow.basedir}/scripts/tabulate_metrics.py"
    output:
        f"results/{results_dir}/{{base_config}}/ablations_summary.csv"
    params:
        results_dir=results_dir
    group:
        "evaluate"
    shell:
        """
        python {input.script} {input.agg} \
            --output-dir results/{params.results_dir}/{wildcards.base_config} \
            --output-tag ablations_summary
        """