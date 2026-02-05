"""Pure design optimization function."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple
import torch
import logging
from sbi_bax.utils.plot import plot_progress
from tqdm import tqdm
from torch import nn

from sbi_bax.utils.optim import ReplacementPolicy


log = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of design optimization."""

    best_design: torch.Tensor
    all_designs: torch.Tensor
    final_eig: torch.Tensor


def per_row_clip_hook(n, max_norm=1.0, eps=1e-8):
    def hook(grad):
        g = grad.view(n, -1)
        norms = g.norm(dim=1, keepdim=True).clamp_min(eps)
        scale = (max_norm / norms).clamp(max=1.0)
        return (g * scale).view_as(grad)

    return hook


def optimize_designs(
    design_set: torch.Tensor,
    compute_loss_fn: Callable[[torch.Tensor, tuple], torch.Tensor],
    n_designs: int,
    n_steps: int,
    burn_in: int,
    lr: float,
    workdir: Path,
    step_optimizers: Callable[[torch.Tensor], None] | None = None,
    steps_plot: int = 50,
    eval_samples: int = 100,
    device: str = "cpu",
    eig_plot: Callable[[torch.Tensor, torch.Tensor, Path], None] | None = None,
    bounds: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # (low, high) broadcastable to designs
    replacement_policy: Optional[ReplacementPolicy] = None,
    additional_cost: Optional[Callable[[torch.Tensor, int, dict], torch.Tensor]] = None,
    max_norm: float = 1.0,
    repeat_n_final: int = 10,
    save_final_designs: bool = False,
) -> OptimizationResult:
    """
    Pure design optimization loop.

    Args:
        design_set: Pool of candidate designs to sample from
        get_training_data_fn: Function that generates training data for given designs
        update_models_fn: Function that updates models, returns dict of losses
        compute_loss_fn: Function that computes loss (negative EIG) for designs
        n_designs: Number of candidate designs to optimize
        n_steps: Number of optimization steps
        burn_in: Steps before starting design optimization
        lr: Learning rate for design optimizer
        workdir: Directory to save results
        steps_plot: Plot every N steps
        device: Device to run on
        eig_plot: Optional function to plot EIG landscape
        bounds: Optional (low, high) bounds for box constraints on designs

    Returns:
        OptimizationResult with best design and diagnostics
    """
    workdir.mkdir(exist_ok=True, parents=True)

    # Initialize designs randomly from design set
    candidate_designs = nn.Parameter(
        design_set[torch.randint(0, design_set.shape[0], (n_designs,))].to(device)
    )

    # Per-design grad clipping on a single tensor
    candidate_designs.register_hook(per_row_clip_hook(n_designs, max_norm=max_norm))
    # Expand bounds for clamping
    if bounds is not None:
        low, high = (b.to(device).expand_as(candidate_designs) for b in bounds)

    # Setup design optimizer, use RMSprop for MCMC loss landscape
    design_optimizer = torch.optim.RMSprop([candidate_designs], lr=lr)

    history = []
    best_eig = []
    all_logs = defaultdict(list)

    for step in tqdm(range(n_steps), desc="Optimizing designs"):
        # Detach the designs during burn-in
        designs_view = (
            candidate_designs.detach() if step < burn_in else candidate_designs
        )
        # Compute loss and logs
        full_loss, step_logs = compute_loss_fn(designs_view)
        # Get sum loss over designs (don't divide grads for each design!)
        loss = full_loss.sum()
        # Update designs
        if step > burn_in:
            # Add additional cost if provided
            if additional_cost is not None:
                cost = additional_cost(candidate_designs, step - burn_in, all_logs)
                loss += cost.sum()
            # Zero gradients
            design_optimizer.zero_grad()
            # Backward and step
            loss.backward()
            with torch.no_grad():
                design_optimizer.step()
            # Project back into the box (clamp)
            if bounds is not None:
                with torch.no_grad():
                    candidate_designs.clamp_(low, high)
            # Apply replacement policy
            if replacement_policy is not None:
                repl_logs = replacement_policy.maybe_replace(
                    candidate_designs,
                    bounds=bounds,
                    per_design_loss=full_loss.detach(),
                    step=step,
                )
                all_logs["n_replaced"].append(repl_logs.get("n_replaced", 0))
        # Step any additional optimizers
        if step_optimizers is not None:
            step_optimizers()
        # Record stats
        history.append(full_loss.mean().item())
        best_eig.append(full_loss.min().item())
        # Collect logs
        for key, val in step_logs.items():
            all_logs[key].extend(val)

        # Plot progress
        if (step + 1) % steps_plot == 0 or (step == n_steps - 1):
            plot_progress(history, best_eig, all_logs, workdir / "losses.png")
            if eig_plot is not None:
                eig_plot(
                    candidate_designs.detach().cpu(),
                    -full_loss.detach().cpu(),
                    workdir / f"eig_step_{step + 1}.png",
                )

    # Compute final EIG
    with torch.no_grad():
        # Compute final EIG
        final_eig = torch.zeros(n_designs, device=device)
        for _ in range(repeat_n_final):
            final_eig -= compute_loss_fn(
                candidate_designs, n_samples=eval_samples, is_final=True
            )[0]
        if eig_plot is not None:
            eig_plot(
                candidate_designs.detach().cpu(),
                final_eig.detach().cpu(),
                workdir / "eig_step_final.png",
            )
        if save_final_designs:
            torch.save(
                {
                    "designs": candidate_designs.detach().cpu(),
                    "final_eig": final_eig.detach().cpu() / repeat_n_final,
                },
                workdir / "final_designs.pt",
            )

    # Get best design
    best_idx = torch.argmax(final_eig)
    return OptimizationResult(
        best_design=candidate_designs[best_idx].detach().cpu(),
        all_designs=candidate_designs.detach().cpu(),
        final_eig=final_eig.detach().cpu(),
    )
