import torch
from typing import Sequence, Optional


class LinearWarmupLRScheduler:
    """
    Linearly warm up selected optimizer param groups from 0 → target_lr over warmup_steps.
    If group_indices is None, applies to all param groups.
    If warmup_steps <= 0, this is a no-op.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        group_indices: Optional[Sequence[int]] = None,
        warmup_steps: int = 0,
    ):
        self.optimizer = optimizer
        self.group_indices = (
            list(range(len(optimizer.param_groups)))
            if group_indices is None
            else list(group_indices)
        )
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0

        # Store target LRs
        self.target_lrs = {
            i: optimizer.param_groups[i]["lr"] for i in self.group_indices
        }

        # No-op if no warmup
        if self.warmup_steps <= 0:
            return

        # Set selected groups to 0 initially
        for i in self.group_indices:
            optimizer.param_groups[i]["lr"] = 0.0

    def step(self):
        if self.warmup_steps <= 0:
            return
        if self.step_count >= self.warmup_steps:
            # Ensure final target LR
            for i, lr in self.target_lrs.items():
                self.optimizer.param_groups[i]["lr"] = lr
            return
        scale = (self.step_count + 1) / float(self.warmup_steps)
        for i, lr in self.target_lrs.items():
            self.optimizer.param_groups[i]["lr"] = lr * scale
        self.step_count += 1
