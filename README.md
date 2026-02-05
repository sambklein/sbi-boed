# SBI-BAX: Simulation-Based Inference for Bayesian Algorithm Execution

This repository implements sequential Bayesian optimal experimental design (BOED) using simulation-based inference (SBI) techniques. The codebase supports multiple acquisition strategies including Neural Posterior Estimation (NPE), Neural Likelihood Estimation (NLE), and contrastive methods for design optimization across various benchmark problems.

## About This Codebase

**Note on Code Organization**: This codebase was originally developed as a research prototype for exploratory experimentation and has evolved organically over the course of the project. While functional and capable of reproducing all results in our paper, the code organization reflects its experimental origins rather than production software engineering standards. We have prioritized research flexibility and rapid iteration over architectural purity. Users should expect some inconsistencies in naming conventions, duplicated functionality across modules, and experimental code paths that remain in the repository. We appreciate your understanding and welcome contributions that improve code quality while maintaining reproducibility.

## Environment Setup

### Prerequisites

1. **Install Pixi** (recommended package manager): Follow the [Pixi installation guide](https://pixi.sh/latest/installation/)
2. Clone this repository:
   ```bash
   git clone https://github.com/sambklein/sbi-bax.git
   cd sbi-bax
   ```

### Creating the Environment

**For CPU-only systems (macOS, Linux without CUDA):**
```bash
pixi shell
```

**For GPU-enabled Linux systems (recommended for faster training):**
```bash
pixi shell -e cuda
```

This will automatically install all dependencies and place you in a shell where the code in this repository can be run. The CUDA environment will configure PyTorch with CUDA support for GPU acceleration.

## Project Structure

```
sbi-bax/
├── src/sbi_bax/          # Main package source
│   ├── data/             # Dataset classes and priors
│   ├── experiments/      # Acquisition strategies (NPE, NLE, contrastive)
│   ├── models/           # Neural network architectures
│   ├── simulators/       # Forward models for benchmarks
│   └── utils/            # Utility functions
├── scripts/              # Executable scripts for experiments
├── config/               # Hydra configuration files
├── Snakefile            # Main experiment workflow
├── ablations.smk        # Ablation study workflow
└── notebooks/           # Analysis and visualization notebooks
```

## Running Experiments

The repository provides two main workflows controlled by Snakemake files. Experiments can be run either through Snakemake (recommended for batch processing) or directly via Python scripts (useful for debugging or single runs).

### Main Experiments (Snakefile)

The primary workflow runs sequential design experiments across multiple benchmarks and acquisition strategies.

**Using Snakemake (recommended):**
```bash
# Run the n_designs benchmark for all measurements (makes Figure 2)
snakemake --configfile ./config/paper_compare_n_designs.yaml --profile config/profiles/s3df -s ablations.smk

# Run all (posterior-eig_estimator) NLE-NLE, NPE-NPE, NRE-NRE experiments on source finding in 2D, 3D and 5D
pixi run -e cuda snakemake --configfile ./config/paper_many_experiments.yaml --profile config/profiles/s3df

# Run all NPE-NRE experiments on all benchmarks except CES
snakemake --configfile ./config/paper_performance.yaml --profile config/profiles/s3df

# Run all NPE-NRE experiments on CES
snakemake --configfile ./config/paper_ces.yaml --profile config/profiles/ampere
```

**Available configurations:**
- `source_finding`: Source localization benchmark
- `pharmacokinetic`: Pharmacokinetic parameter estimation
- `ces`: Constant elasticity of substitution preference model

**Available acquisition strategies:**
- `npe`: Neural Posterior Estimation
- `nle`: Neural Likelihood Estimation  
- `contrastive`: Contrastive MI estimation
- `npe_contrastive`: NPE posterior with contrastive acquisition
- `nle_contrastive`: NLE posterior with contrastive acquisition

## Configuration System

Experiments are configured using [Hydra](https://hydra.cc/), allowing flexible parameter overrides via command line. Key configuration groups include:

- **Base configs** (`config/*.yaml`): Problem-specific settings (simulators, priors, design spaces)
- **Acquisition** (`config/acquisition/*.yaml`): Acquisition strategy hyperparameters
- **SBI** (`config/sbi/*.yaml`): Neural network architectures and training settings
- **Optimization** (`config/outer_optimization/*.yaml`): Design optimization hyperparameters

**Example parameter overrides:**
```bash
python scripts/run_functional.py \
    --config-name source_finding \
    seed=123 \
    acquisition=npe \
    sbi.n_thetas=50000 \
    outer_optimization.n_iter=100 \
    data.n_measure_init=3
```

## Cluster Execution (SLURM)

For large-scale experiments on HPC clusters with SLURM:

```bash
# Configure SLURM profile in config/profiles/slurm/
pixi run -e cuda snakemake -s Snakefile \
    --profile config/profiles/slurm \
    --jobs 50 \
    --executor slurm
```

The SLURM profile configures job submission parameters (time limits, memory, GPU requirements) defined in `config/profiles/slurm/config.yaml`.

## Visualization and Analysis

Posterior visualization and design optimization results can be generated using:

```bash
python scripts/plot_posterior_and_opt.py
```

This produces publication-quality plots in `results/paper_plots/` including:
- Posterior distributions for each parameter
- Design optimization landscapes with EIG values
- Source localization visualizations

## Citation

If you use this code in your research, please cite the original paper.

<!-- TODO: put in proper citation  -->

## Resources

- [Pixi Documentation](https://pixi.sh/)
- [Hydra Configuration Framework](https://hydra.cc/)
- [SBI Library](https://www.mackelab.org/sbi/)
- [Snakemake Workflow Management](https://snakemake.readthedocs.io/)

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.