# SB+ BAX for crystal diffraction images
import logging
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import torch
from nflows import distributions
import torch.nn as nn
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.inference import NRE_A
import pickle

from sbi_bax.models.conditional_linear import ConditionalDiagonalLinear
from sbi_bax.models.flow import NuisanceFlow, spline_inn
from sbi_bax.models.coord_transform import BatchedCartesianToHypersphericalTransform
from sbi_bax.utils.torch_utils import get_device
from nflows.transforms import PointwiseAffineTransform
from torch.utils.data import DataLoader, TensorDataset
from nflows.transforms.base import CompositeTransform

import numpy as np
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior

from sbi_bax.models.flow import GeneralFlow
from sbi_bax.models.mlp import Mlp
from sbi_bax.utils.torch_utils import sample_theta
from sbi_bax.utils.train_flow import train_flow

# Set up logging
log = logging.getLogger(__name__)


def data_ea_likelihood(ea_posterior):
    def likelihood_fn(x, ea_path):
        ea_posterior.set_default_x(ea_path)
        # Evaluate the posterior at the given x and ea_path
        return ea_posterior.log_prob(x, norm_posterior=False).exp()

    return likelihood_fn


def check_bound(samples, bounds):
    """
    Check if samples are within the specified bounds.
    """
    return ((samples >= bounds[0]) & (samples <= bounds[1])).all()


def is_in_bounds(samples, bounds):
    """
    Check if samples are within the specified bounds.
    """
    for i in range(samples.shape[1]):
        if not check_bound(samples[:, i], bounds[i]):
            log.warning(
                f"Sample is out of bounds for parameter {i} with bounds {bounds[i]}"
                f" max {samples[:, i].max():.2f} min {samples[:, i].min():.2f}"
            )


def plot_posterior(theta_estimates, prior_bounds, true_angles, measure_dir):
    # If the samples exceed the prior bounds return a warning
    is_in_bounds(theta_estimates, prior_bounds)

    # Plot the posterior estimates
    pairplot(
        theta_estimates.numpy(),
        points=true_angles.cpu().numpy(),
        limits=prior_bounds,
        figsize=(6, 6),
    )
    plt.savefig(measure_dir / "posterior_theta_sbi.png", bbox_inches="tight")
    plt.close()


def sample_with_rejection(
    posterior,
    data_obs,
    prior_bounds,
    n_samples,
    max_attempts=10,
    default_method=True,
    prior_constraint=None,
):
    """
    Sample from posterior with rejection sampling to ensure all samples are within prior bounds.

    Args:
        posterior: SBI posterior object
        data_obs: Observed data tensor
        prior_bounds: List of (low, high) tuples for each parameter
        n_samples: Desired number of samples
        max_attempts: Maximum number of resampling attempts

    Returns:
        theta_estimates: Tensor of shape (n_samples, n_params) with all samples in bounds
    """
    device = posterior.posterior_estimator.device

    # Ensure data_obs has batch dimension
    if len(data_obs.shape) == 2:
        data_obs = data_obs.unsqueeze(0)
    data_obs = data_obs.to(device)

    # Convert bounds to tensor for vectorized checking
    bounds_low = torch.tensor([b[0] for b in prior_bounds], device=device)
    bounds_high = torch.tensor([b[1] for b in prior_bounds], device=device)

    all_samples = []
    n_needed = n_samples
    attempt = 0

    while n_needed > 0 and attempt < max_attempts:
        # Sample more than needed to account for rejections (oversample by 50%)
        n_to_sample = int(n_needed * 1.5) if attempt > 0 else n_needed

        # Generate samples
        with torch.no_grad():
            samples = (
                posterior.posterior_estimator.net.sample(n_to_sample, data_obs)
                .cpu()
                .squeeze(0)
            )

        # Check which samples are in bounds
        in_bounds = (
            (samples >= bounds_low.cpu()) & (samples <= bounds_high.cpu())
        ).all(dim=-1)

        if prior_constraint is not None:
            constraint_mask = prior_constraint(samples)
            in_bounds = in_bounds & constraint_mask.cpu()

        # Keep only valid samples
        valid_samples = samples[in_bounds]
        all_samples.append(valid_samples[:n_needed])  # Don't take more than needed

        n_accepted = len(valid_samples[:n_needed])
        n_needed -= n_accepted
        attempt += 1

        if n_needed > 0:
            log.warning(
                f"Attempt {attempt}: Accepted {n_accepted}/{n_to_sample} samples. "
                f"Still need {n_needed} more samples."
            )

    if n_needed > 0:
        log.error(
            f"Could not generate {n_samples} valid samples after {max_attempts} attempts. "
            f"Only got {n_samples - n_needed} valid samples. "
            f"Consider checking if posterior has significant mass outside prior bounds."
        )
        # Return what we have
        return (
            torch.cat(all_samples, dim=0)
            if all_samples
            else torch.empty(0, len(prior_bounds))
        )

    # Concatenate all samples and return exactly n_samples
    theta_estimates = torch.cat(all_samples, dim=0)[:n_samples]

    log.info(
        f"Successfully generated {n_samples} samples within prior bounds "
        f"after {attempt} attempt(s)."
    )

    return theta_estimates


def plot_sbi_training_loss(inference, measure_dir):
    train_figure = measure_dir / "sbi_training_loss.png"
    train_figure.parent.mkdir(parents=True, exist_ok=True)
    train_loss = inference._summary["training_loss"]
    val_loss = inference._summary["validation_loss"]
    fig, ax = plt.subplots(figsize=(10, 6))
    # Add a grid
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.plot(train_loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.savefig(train_figure, bbox_inches="tight")
    plt.close(fig)


def run_sbi(
    data: torch.Tensor,
    thetas: torch.Tensor,
    prior: BoxUniform,
    embedding_net: nn.Module,
    sbi_bs: int = 512,
    sbi_lr: float = 1e-3,
    max_epochs: int = 1000,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.2,
    measure_dir: Path = Path("measurements"),
    prior_bounds: list = [0, 1],
    simulator=None,
    true_angles: torch.Tensor = None,
    data_obs: torch.Tensor = None,
    norm_theta: str = None,
    from_scratch: bool = True,
    model: str = "mdn",
    initial_model: Path = None,  # NOTE: not using this anymore!
    n_samples: int = 100_000,
    hyperspherical: bool = True,
    uniform_base: bool = False,
    prior_constraint: callable = None,
):
    # Make the directory if it doesn't already exist
    measure_dir.mkdir(parents=True, exist_ok=True)
    final_model = measure_dir / "model.pt"
    if from_scratch or not final_model.exists():
        # If from_scratch is True or model does not exist, train a new model
        log.info("Training SBI posterior from scratch")

        # If the embedding net has a norm method, fit it to the data
        if hasattr(embedding_net, "fit_norm"):
            embedding_net.fit_norm(data)
            log.info("Fitted embedding net normalization to data")
        else:
            log.warning(
                "Embedding net does not have a fit_norm method, data may not be normalized"
            )

        # Inlined get_sbi_posterior function contents
        # Define posterior neural network
        def post_nn(batch_theta, batch_obs):
            input_transform_list = [make_norm_layer(thetas, 3.5)]
            if hyperspherical:
                input_transform_list.append(
                    BatchedCartesianToHypersphericalTransform(thetas.shape[-1], 2)
                )
            model = GeneralFlow(
                batch_obs,
                batch_theta,
                embedding_net,
                lambda dim, context: spline_inn(
                    dim,
                    128,
                    nstack=2,
                    num_blocks=1,
                    context_features=context,
                    tails="linear",
                    tail_bound=3.5,
                    input_transform=CompositeTransform(
                        input_transform_list
                        + [ConditionalDiagonalLinear(batch_theta.shape[-1], context)]
                        if not uniform_base
                        else input_transform_list  # Can't use unbounded transforms with uniform base
                    ),
                ),
                uniform_base=uniform_base,
            )
            return model

        # Initialize NPE inference
        inference = NPE(prior, post_nn, device=get_device())

        # Train the model with the data, making sure to shuffle the data
        index = torch.randperm(thetas.shape[0])
        inference.append_simulations(
            thetas[index], data[index], data_device=get_device()
        )

        # Train the inference object
        density_estimator = inference.train(
            max_num_epochs=max_epochs,
            training_batch_size=sbi_bs,
            learning_rate=sbi_lr,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=validation_fraction,
        )

        # Plot some training diagnostics
        plot_sbi_training_loss(inference, measure_dir)

        # Inference object is pickleable according to sbi docs
        with open(final_model, "wb") as f:
            pickle.dump(inference, f)
        log.info(f"Saved SBI posterior model to {final_model}")
    else:
        # If the model exists, load it
        log.info(f"Loading existing SBI posterior model from {final_model}")
        with open(final_model, "rb") as f:
            inference = pickle.load(f)
        # Get the density estimator from the inference object
        density_estimator = inference._neural_net

    # Return the posterior distribution
    posterior = inference.build_posterior(density_estimator)
    # Generate samples from the posterior
    posterior.set_default_x(data_obs)
    # Sample from the posterior
    theta_estimates = sample_with_rejection(
        posterior,
        data_obs,
        prior_bounds,
        n_samples,
        max_attempts=10,
        prior_constraint=prior_constraint,
    )
    # Plot posterior
    plot_posterior(theta_estimates, prior_bounds, true_angles, measure_dir)
    # Save samples to disk
    torch.save(theta_estimates, measure_dir / "samples_post.pt")

    return posterior, theta_estimates


def run_sbi_nre(
    data: torch.Tensor,
    thetas: torch.Tensor,
    prior: BoxUniform,
    classifier: nn.Module,
    sbi_bs: int = 512,
    sbi_lr: float = 1e-3,
    max_epochs: int = 1000,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.2,
    measure_dir: Path = Path("measurements"),
    prior_bounds: list = [0, 1],
    simulator=None,
    true_angles: torch.Tensor = None,
    data_obs: torch.Tensor = None,
    norm_theta: str = None,
    from_scratch: bool = True,
    model: str = "mdn",
    initial_model: Path = None,  # NOTE: not using this anymore!
    n_samples: int = 100_000,
    hyperspherical: bool = True,
):
    # Similar to run_sbi but using NRE instead of NPE (NOTE: code just duplicated for now)
    measure_dir.mkdir(parents=True, exist_ok=True)
    final_model = measure_dir / "model.pt"
    if from_scratch or not final_model.exists():
        # If from_scratch is True or model does not exist, train a new model
        log.info("Training SBI posterior from scratch")

        # If the embedding net has a norm method, fit it to the data
        if hasattr(classifier, "fit_norm"):
            classifier.fit_norm(thetas, data)
            log.info("Fitted embedding net normalization to data")
        else:
            log.warning(
                "Embedding net does not have a fit_norm method, data may not be normalized"
            )

        # Initialize NPE inference
        inference = NRE_A(prior, device=get_device())
        # Skip internal model building, use our own
        inference._neural_net = classifier

        # Train the model with the data, making sure to shuffle the data
        index = torch.randperm(thetas.shape[0])
        inference.append_simulations(
            thetas[index], data[index], data_device=get_device()
        )

        # Train the inference object
        density_estimator = inference.train(
            max_num_epochs=max_epochs,
            training_batch_size=sbi_bs,
            learning_rate=sbi_lr,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=validation_fraction,
        )

        # Plot some training diagnostics
        plot_sbi_training_loss(inference, measure_dir)

        # Inference object is pickleable according to sbi docs
        with open(final_model, "wb") as f:
            pickle.dump(inference, f)
        log.info(f"Saved SBI posterior model to {final_model}")
    else:
        # If the model exists, load it
        log.info(f"Loading existing SBI posterior model from {final_model}")
        with open(final_model, "rb") as f:
            inference = pickle.load(f)
        # Get the density estimator from the inference object
        density_estimator = inference._neural_net

    # Return the posterior distribution
    posterior = inference.build_posterior(density_estimator)
    # Generate samples from the posterior
    posterior.set_default_x(data_obs)
    theta_estimates = sample_theta(
        posterior, measure_dir / "samples_post.pt", from_scratch, n_samples
    )
    plot_posterior(theta_estimates, prior_bounds, true_angles, measure_dir)

    return posterior, theta_estimates


def make_norm_layer(data, bound, inverse=False):
    # Define a layer that will transform data to [-bound, bound]
    d_min = data.min(axis=0)[0]
    d_max = data.max(axis=0)[0]
    denom = d_max - d_min
    scale = 2 * bound / denom
    shift = scale * d_min + bound
    if inverse:
        # return the inverse transform
        return PointwiseAffineTransform(scale=1 / scale, shift=shift / scale)
    else:
        return PointwiseAffineTransform(scale=scale, shift=-shift)


def train_nuisance_flow(
    data: torch.Tensor,
    nuisance: torch.Tensor,
    thetas: torch.Tensor,
    embeddor: nn.Module,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.2,
    measure_dir: Path = Path("measurements"),
    from_scratch: bool = True,
    **kwargs,
):
    # Make the directory if it doesn't already exist
    measure_dir.mkdir(parents=True, exist_ok=True)
    final_model = measure_dir / "model.pt"
    # Create the flow model
    dim = nuisance.shape[1]
    theta_embeddor = nn.Sequential(
        nn.Linear(thetas.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )
    n_context = embeddor(data[:10]).shape[1] + theta_embeddor(thetas).shape[1]
    # If the embedding net has a norm method, fit it to the data
    if hasattr(embeddor, "fit_norm"):
        embeddor.fit_norm(data)
        log.info("Fitted embedding net normalization to data")
    else:
        log.warning(
            "Embedding net does not have a fit_norm method, data may not be normalized"
        )

    # Extract the min and max from the nuisance samples
    transform = spline_inn(
        dim,
        128,
        context_features=n_context,
        tails="linear",
        tail_bound=3.5,
        input_transform=make_norm_layer(nuisance, 3.5),
    )
    flow = NuisanceFlow(
        transform,
        distributions.StandardNormal([nuisance.shape[1]]),
        embeddor,
        theta_embeddor,
    )
    if from_scratch or not final_model.exists():
        # Prepare dataset and dataloader
        dataset = TensorDataset(data, thetas, nuisance)
        n_val = int(len(dataset) * validation_fraction)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)

        # Learning rate schedule: warmup, constant, linear ramp down
        def get_lr(epoch):
            warmup_epochs = int(0.05 * max_epochs)
            rampdown_epochs = int(0.1 * max_epochs)
            if epoch < warmup_epochs:
                return lr * (epoch + 1) / warmup_epochs
            elif epoch < max_epochs - rampdown_epochs:
                return lr
            else:
                # Linear ramp down
                return lr * (max_epochs - epoch) / rampdown_epochs

        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(max_epochs):
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = get_lr(epoch)

            flow.train()
            train_loss = 0.0
            for batch_data, batch_theta, batch_nuis in train_loader:
                optimizer.zero_grad()
                # Negative log likelihood loss
                log_prob = flow.log_prob(batch_nuis, batch_theta, batch_data)
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_data.size(0)
            train_loss /= n_train
            train_losses.append(train_loss)

            # Validation
            flow.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_theta, batch_nuis in val_loader:
                    log_prob = flow.log_prob(batch_nuis, batch_theta, batch_data)
                    loss = -log_prob.mean()
                    val_loss += loss.item() * batch_data.size(0)
            val_loss /= n_val
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                torch.save(flow.state_dict(), measure_dir / "best_nuisance_flow.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= stop_after_epochs:
                    print("Early stopping triggered.")
                    break

        # Plot training curve
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Nuisance Flow Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(measure_dir / "nuisance_flow_training.png")
        plt.close()

        # Copy best model to final model location
        shutil.copy2(measure_dir / "best_nuisance_flow.pt", final_model)
    flow.load_state_dict(torch.load(final_model))
    return flow


def arrange_nle_data(
    data, thetas: torch.Tensor, obs_design: torch.Tensor, n_measured: int
):
    """
    Arrange data for NLE training.

    Args:
        data: Data object with split_obs method
        thetas: Theta samples [n_samples, n_params]
        obs_design: Combined observations and designs
        n_measured: Number of measurements taken

    Returns:
        obs: Observations [n_samples * n_measured, obs_dim]
        theta_designs: Concatenated thetas and designs [n_samples * n_measured, n_params + design_dim]
    """
    obs, designs = data.split_obs(obs_design, n_measured)
    if obs.ndim == 3:
        obs = obs.squeeze(-1)

    n_first = thetas.shape[0] * n_measured
    designs = designs.view(n_first, designs.shape[-1])
    thetas = thetas.repeat_interleave(n_measured, 0)
    if designs.ndim > 2:
        designs = designs.squeeze()
    theta_designs = torch.cat((thetas, designs), -1)

    if obs.ndim == 1 or obs.shape[1] == n_measured:
        obs = obs.view(-1, 1)

    return obs, theta_designs


def build_nle_likelihood_flow(
    theta_designs: torch.Tensor,
    obs: torch.Tensor,
    encode_dim: int = 128,
) -> GeneralFlow:
    """
    Build likelihood flow p(y | θ, ξ) for NLE.

    Args:
        theta_designs: Concatenated thetas and designs
        obs: Observations
        encode_dim: Encoding dimension for context network

    Returns:
        GeneralFlow object (not yet trained)
    """
    flow = GeneralFlow(
        theta_designs.unsqueeze(1),
        obs,
        Mlp(theta_designs.shape[-1], encode_dim, hidden_dims=[]),
        lambda dim, context: spline_inn(
            dim,
            128,
            nstack=2,
            context_features=context,
            tails="linear",
            tail_bound=3.5,
            num_blocks=1,
            # Normalize the data and also allow for learning a context dependent shift and scale
            input_transform=CompositeTransform(
                [make_norm_layer(obs, 3.5), ConditionalDiagonalLinear(dim, context)]
            ),
        ),
    )
    flow.net._embedding_net.fit_norm(theta_designs)
    return flow


def train_nle_likelihood_flow(
    flow_net,
    obs: torch.Tensor,
    theta_designs: torch.Tensor,
    measure_dir: Path,
    initial_model: Path,
    sbi_params: dict,
    device: torch.device,
):
    """
    Train the NLE likelihood flow.

    Args:
        flow_net: Flow network to train
        obs: Observations
        theta_designs: Concatenated thetas and designs
        measure_dir: Directory for saving results
        initial_model: Path to previous model for warm start
        sbi_params: SBI training parameters
        device: Device to train on

    Returns:
        Trained flow network
    """
    workdir = measure_dir / "nle_likelihood"
    workdir.mkdir(exist_ok=True, parents=True)

    # Check for previous model
    if initial_model is not None and initial_model.exists():
        prev_model = initial_model.parent / "nle_likelihood" / "best_flow.pt"
        if not prev_model.exists():
            prev_model = None
    else:
        prev_model = None

    flow_net.to(device)

    return train_flow(
        flow_net,
        obs.to(device),
        theta_designs.to(device),
        work_dir=workdir,
        initial_model=prev_model,
        from_scratch=sbi_params.get("from_scratch", False),
        batch_size=sbi_params["sbi_bs"],
        lr=sbi_params["sbi_lr"],
        stop_after_epochs=sbi_params["stop_after_epochs"],
        validation_fraction=sbi_params["validation_fraction"],
        max_epochs=sbi_params["max_epochs"],
        max_batches=np.ceil(1_000_000 / sbi_params["sbi_bs"]).astype(int),
    )


def build_nle_mcmc_posterior(
    flow: GeneralFlow,
    prior,
    data,
    n_measured: int,
    data_obs: torch.Tensor,
    measure_dir: Path,
    sbi_params: dict,
    n_samples: int,
):
    """
    Build MCMC posterior for NLE using trained likelihood flow.

    Args:
        flow: Trained likelihood flow
        prior: Prior distribution
        data: Data object with arrange methods
        n_measured: Number of measurements taken
        data_obs: Observed data
        measure_dir: Directory for saving samples
        sbi_params: SBI parameters (for from_scratch flag)

    Returns:
        (posterior, theta_samples) tuple
    """

    # Build potential function
    def potential(theta, x0):
        device = theta.device
        obs, theta_designs = arrange_nle_data(
            data,
            theta,
            x0.repeat_interleave(len(theta), 0).to(device),
            n_measured,
        )
        flow.to(device)
        lp = flow.net.log_prob(obs, context=theta_designs).view(-1, n_measured).sum(-1)
        pl = prior.log_prob(theta.to(get_device())).to(device)
        return lp + pl

    # Create MCMC posterior
    posterior = MCMCPosterior(potential, prior)
    posterior.set_default_x(data_obs.unsqueeze(0) if data_obs.ndim == 2 else data_obs)

    # Sample from posterior
    theta_estimates = sample_theta(
        posterior,
        measure_dir / "samples_post.pt",
        sbi_params.get("from_scratch", False),
        n_samples=n_samples,
    )

    # Plot posterior
    plot_posterior(
        theta_estimates,
        list(zip(data.prior_low, data.prior_high)),
        data.true_thetas,
        measure_dir,
    )

    return posterior, theta_estimates


def setup_nle_posterior(
    train_obs: torch.Tensor,
    train_x: torch.Tensor,
    thetas: torch.Tensor,
    prior,
    n_measured: int,
    data_obs: torch.Tensor,
    measure_dir: Path,
    initial_model: Path,
    sbi_params: dict,
    data,
    n_samples: int,
    device: torch.device,
):
    """
    Complete NLE posterior setup: train likelihood flow and build MCMC posterior.

    This is the main entry point that orchestrates all NLE posterior construction.

    Args:
        train_obs: Training observations
        train_x: Training designs
        thetas: Theta samples
        prior: Prior distribution
        n_measured: Number of measurements taken
        data_obs: Observed data
        measure_dir: Directory for saving results
        initial_model: Path to previous model for warm start
        sbi_params: SBI training parameters
        data: Data object
        device: Device to train on

    Returns:
        (posterior, theta_samples) tuple
    """
    # 1. Combine and arrange data
    obs_design = data.combine_no_ea(train_obs, train_x, n_measured)
    obs, theta_designs = arrange_nle_data(data, thetas, obs_design, n_measured)

    # 2. Build likelihood flow
    flow = build_nle_likelihood_flow(theta_designs, obs)

    # 3. Train likelihood flow
    flow.net = train_nle_likelihood_flow(
        flow.net,
        obs,
        theta_designs,
        measure_dir,
        initial_model,
        sbi_params,
        device,
    )

    # 4. Build MCMC posterior and sample
    return build_nle_mcmc_posterior(
        flow, prior, data, n_measured, data_obs, measure_dir, sbi_params, n_samples
    )
