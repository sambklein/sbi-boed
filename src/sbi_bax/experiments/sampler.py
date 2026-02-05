import torch
import numpy as np
from pathlib import Path
from typing import Callable, Tuple, Optional


def langevin_sample(
    potential_net,
    n_samples: int,
    n_steps: int = 100,
    step_size: float = 0.01,
    design_dim: int = None,
    bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    temperature: float = 1.0,
    device: str = "cuda",
    init_samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample designs using Langevin dynamics.

    Args:
        potential_net: Network that predicts potential(ξ) ∝ loss(ξ)
        n_samples: Number of samples to generate
        n_steps: Number of Langevin steps
        step_size: Step size for Langevin updates
        design_dim: Dimension of design space
        bounds: (low, high) tuple for box constraints
        temperature: Temperature for sampling (higher = more noise)
        device: Device to run on
        init_samples: Optional initial samples to refine

    Returns:
        Samples from p(ξ) ∝ exp(-potential(ξ)/temperature)
    """
    # Initialize samples
    if init_samples is not None:
        designs = init_samples.clone().detach().to(device)
    elif bounds is not None:
        low, high = bounds
        designs = torch.rand(n_samples, design_dim, device=device) * (high - low) + low
    else:
        designs = torch.randn(n_samples, design_dim, device=device)

    designs.requires_grad_(True)

    for step in range(n_steps):
        # Compute potential and its gradient
        potential = potential_net(designs)

        # Score: ∇_ξ log p(ξ) = -∇_ξ potential(ξ) / temperature
        score = torch.autograd.grad(potential.sum(), designs, create_graph=False)[0]

        with torch.no_grad():
            # Langevin update: ξ_{t+1} = ξ_t - ε/2 * ∇potential + √ε * noise
            noise = torch.randn_like(designs) * np.sqrt(temperature)
            designs.data = (
                designs.data - (step_size / 2) * score + np.sqrt(step_size) * noise
            )

            # Project back into bounds
            if bounds is not None:
                low, high = bounds
                designs.data.clamp_(low, high)

    return designs.detach()


def optimize_designs_with_langevin(
    potential_net,
    get_training_data_fn: Callable,
    update_models_fn: Callable,
    compute_loss_fn: Callable,
    plot_eig_fn: Optional[Callable] = None,
    n_steps: int = 1000,
    burn_in: int = 100,
    n_designs: int = 128,
    design_dim: int = 2,
    bounds: Tuple[float, float] = (-3.5, 3.5),
    temperature: float = 10.0,
    langevin_steps: int = 50,
    langevin_step_size: float = 0.01,
    n_mc: int = 512,
    potential_train_freq: int = 10,
    potential_epochs: int = 5,
    plot_freq: int = 100,
    workdir: Optional[Path] = None,
    device: str = "cuda",
    log_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Optimize experimental designs using Langevin dynamics with a learned potential.

    Args:
        potential_net: Neural network that predicts potential(ξ) ∝ loss(ξ)
        get_training_data_fn: Function(designs, n_samples) -> training_data
        update_models_fn: Function(training_data) -> None (updates base_flow, top_flow)
        compute_loss_fn: Function(designs, n_samples, is_final) -> (loss, info)
        plot_eig_fn: Optional function(designs, eig, filepath) -> None
        n_steps: Total optimization steps
        burn_in: Number of burn-in steps (if needed)
        n_designs: Number of designs to sample per step
        design_dim: Dimension of design space
        bounds: (low, high) bounds for design space
        temperature: Sampling temperature
        langevin_steps: Number of Langevin steps per sample
        langevin_step_size: Step size for Langevin updates
        n_mc: Number of MC samples for loss computation
        potential_train_freq: Train potential network every N steps
        potential_epochs: Epochs to train potential network
        plot_freq: Plot every N steps
        workdir: Directory for saving plots
        device: Device to run on
        log_fn: Optional logging function

    Returns:
        Best design found (torch.Tensor of shape [design_dim])
    """

    # Setup logging
    def log(msg):
        if log_fn is not None:
            log_fn(msg)
        else:
            print(msg)

    # Setup bounds
    bounds_tensor = (
        torch.tensor([bounds[0]] * design_dim, device=device),
        torch.tensor([bounds[1]] * design_dim, device=device),
    )

    # Main optimization loop
    for step in range(burn_in, n_steps):
        # ===== SAMPLE USING LANGEVIN DYNAMICS =====
        design_samples = langevin_sample(
            potential_net,
            n_samples=n_designs,
            n_steps=langevin_steps,
            step_size=langevin_step_size,
            design_dim=design_dim,
            bounds=bounds_tensor,
            temperature=temperature,
            device=device,
        )

        # ===== GET TRAINING DATA AND UPDATE MODELS =====
        training_data = get_training_data_fn(design_samples, n_mc)
        update_models_fn(training_data)

        # ===== COMPUTE TRUE LOSS =====
        loss, _ = compute_loss_fn(design_samples, n_mc, is_final=False)

        # ===== UPDATE POTENTIAL NETWORK =====
        potential_net.add_samples(design_samples, loss)
        if step % potential_train_freq == 0:
            potential_loss = potential_net.train_on_buffer(n_epochs=potential_epochs)
            log(f"Step {step}: Potential network MSE = {potential_loss:.4f}")

        # ===== PLOTTING =====
        if plot_eig_fn is not None and (step + 1) % plot_freq == 0:
            plot_eig_fn(
                design_samples.detach().cpu(),
                -loss.detach().cpu(),
                workdir / f"eig_step_{step + 1}.png" if workdir else None,
            )

    # ===== FINAL SELECTION =====
    log("Running final Langevin chain for best design selection...")
    with torch.no_grad():
        final_samples = langevin_sample(
            potential_net,
            n_samples=n_designs * 10,
            n_steps=200,  # Longer burn-in for final samples
            step_size=0.005,
            design_dim=design_dim,
            bounds=bounds_tensor,
            temperature=1.0,  # Low temperature for exploitation
            device=device,
        )

        # Evaluate potential on final samples
        final_loss = potential_net(final_samples)
        best_idx = torch.argmin(final_loss)
        best_design = final_samples[best_idx].cpu()

        log(f"Best design selected with potential = {final_loss[best_idx].item():.4f}")

    return best_design
