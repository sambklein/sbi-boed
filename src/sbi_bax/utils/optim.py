import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

log = logging.getLogger(__name__)


def decrease_counter(counter: torch.Tensor, not_stuck: torch.Tensor) -> torch.Tensor:
    """Decrease counter by 1 where not_stuck is True, clamp at 0."""
    return torch.where(
        not_stuck,
        torch.clamp(counter - 1, min=0),
        counter,
    )


def _normalize_to_box(
    designs: torch.Tensor, bounds, eps: float = 1e-12
) -> torch.Tensor:
    if bounds is None:
        return designs
    low, high = bounds
    low = low.to(designs.device).expand_as(designs)
    high = high.to(designs.device).expand_as(designs)
    return (designs - low) / (high - low + eps)


def _hinge_pairwise_penalty(
    z: torch.Tensor, min_sep: float, squared: bool = True
) -> torch.Tensor:
    """
    Squared hinge penalty on pairwise distances: max(0, min_sep - d_ij)^{1 or 2}, averaged over pairs.
    z: [n_designs, ...] (flattened per design internally)
    """
    n = z.shape[0]
    if n < 2:
        return torch.zeros([], device=z.device)
    z_flat = z.view(n, -1)
    dists = torch.cdist(z_flat, z_flat, p=2)  # [n, n]
    i, j = torch.triu_indices(n, n, offset=1)
    dij = dists[i, j]
    pen = F.relu(min_sep - dij)
    pen = pen.pow(2) if squared else pen
    return pen.sum() if pen.numel() > 0 else torch.zeros([], device=z.device)


class PairwiseHingeDiversity:
    """
    Mostly stateless diversity penalty with linear annealing.
    weight(step) = coef * max(0, 1 - step / anneal_steps).
    Optionally normalizes designs to [0,1] box using bounds.
    """

    def __init__(
        self,
        min_sep: float = 0.1,
        coef: float = 0.1,
        anneal_steps: int = 0,
        normalize: bool = True,
        squared: bool = True,
        eps: float = 1e-12,
        bounds: tuple | None = None,
    ):
        self.min_sep = float(min_sep)
        self.coef = float(coef)
        self.anneal_steps = int(anneal_steps)
        self.normalize = bool(normalize)
        self.squared = bool(squared)
        self.eps = float(eps)
        self.bounds = bounds

    def weight(self, step: int) -> float:
        if self.anneal_steps <= 0:
            return 0.0
        decay = 1.0 if step < self.anneal_steps else 0.0
        return self.coef * decay

    def __call__(self, designs: torch.Tensor, step: int, logs: dict) -> torch.Tensor:
        if self.coef == 0.0:
            return torch.zeros([], device=designs.device)
        z = (
            _normalize_to_box(designs, self.bounds, self.eps)
            if self.normalize
            else designs
        )
        base = _hinge_pairwise_penalty(z, self.min_sep, squared=self.squared)
        w = self.weight(step)
        # Cost = weight * base
        cost = base * w
        # Log the cost, mutates so don't return
        logs["diversity_cost"].append([cost.mean().item()])
        return cost


def _to_box_norm(
    x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    return (x - low) / (high - low + eps)


def _from_box_norm(
    z: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    return low + z * (high - low)


@torch.no_grad()
def propose_replacements_around_topk(
    designs: torch.Tensor,
    bounds: Tuple[torch.Tensor, torch.Tensor],
    per_design_loss: torch.Tensor,
    to_reinit: torch.Tensor,  # 1D LongTensor of indices to replace
    topk: int,
    radius: float = 0.05,  # fraction of [0,1] box radius
) -> torch.Tensor:
    """
    Propose new design positions for `to_reinit` by sampling uniformly within a ball
    of normalized radius `radius` around randomly chosen members of the current top-k designs.

    Returns:
        x_new: Tensor of shape [len(to_reinit), *design_shape]
    """
    if to_reinit.numel() == 0:
        return torch.empty_like(designs[:0])

    device = designs.device
    low, high = bounds
    low = low.to(device).expand_as(designs)
    high = high.to(device).expand_as(designs)

    # Normalize to [0,1] box
    z = _to_box_norm(designs, low, high)  # [n, ...]
    z_shape = z.shape[1:]
    d = z[0].numel()

    # Select top-k (best = smallest loss)
    k = max(int(topk), 1)
    k = min(k, per_design_loss.numel())
    topk_indices = torch.topk(per_design_loss, k=k, largest=False).indices

    # Random center for each replacement from the top-k set
    center_idx = topk_indices[
        torch.randint(0, topk_indices.numel(), (to_reinit.numel(),), device=device)
    ]

    # Sample offsets uniformly in d-ball of radius `radius`
    dirs = torch.randn(to_reinit.numel(), d, device=device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
    radii = (torch.rand(to_reinit.numel(), 1, device=device).clamp_min(1e-6)) ** (
        1.0 / d
    )
    offsets = dirs * (radius * radii)

    z_centers = z[center_idx].reshape(to_reinit.numel(), d)
    z_new = (z_centers + offsets).clamp(0.0, 1.0).reshape(to_reinit.numel(), *z_shape)

    # Map back to original box
    x_new = _from_box_norm(z_new, low[to_reinit], high[to_reinit])
    return x_new


class VectorEMA:
    """
    Per-index EMA tracker with optional bias correction.
    Usage:
      ema = VectorEMA(size=n_designs, beta=0.9, bias_correction=True)
      ema.update(per_design_loss)        # each step
      metric = ema.value()               # use for ranking (top-k, etc.)
    """

    def __init__(
        self,
        size: int,
        beta: float = 0.9,
        bias_correction: bool = True,
        device=None,
        dtype=torch.float32,
        init_on_first: bool = True,
    ):
        self.beta = float(beta)
        self.bias_correction = bool(bias_correction)
        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self._steps = 0
        self._init_on_first = bool(init_on_first)

    def to(self, device):
        self.buffer = self.buffer.to(device)
        return self

    @property
    def steps(self) -> int:
        return self._steps

    @torch.no_grad()
    def reset(self, idx: Optional[torch.Tensor] = None):
        if idx is None:
            self.buffer.zero_()
            self._steps = 0
        else:
            self.buffer[idx] = self.value().mean()

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self._steps == 0 and self._init_on_first:
            self.buffer.copy_(x)
        else:
            self.buffer.mul_(self.beta).add_(x, alpha=1.0 - self.beta)
        self._steps += 1

    def value(self) -> torch.Tensor:
        if self.bias_correction and self._steps > 0:
            denom = 1.0 - (self.beta**self._steps)
            return self.buffer / denom
        return self.buffer


class ReplacementPolicy:
    """Interface for candidate replacement strategies."""

    def maybe_replace(
        self,
        designs: torch.Tensor,  # [n, ...], nn.Parameter
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
        per_design_loss: Optional[torch.Tensor],  # [n], lower is better
        step: int,
    ) -> Dict[str, Any]:
        """
        Implementations may modify `designs` in-place under no_grad and return logs.
        Return a dict like {'n_replaced': int, 'indices': Tensor} (keys optional).
        """
        return {}


class BoundaryReinitializer(ReplacementPolicy):
    """
    Reinitialize designs that stick to any box boundary for `patience` steps.
    New designs are sampled within a normalized radius around the current best design.
    """

    def __init__(
        self,
        n_designs: int,
        patience: int = 10,
        tol: float = 1e-6,
        radius: float = 0.05,  # fraction in [0,1] box
        max_per_step: int = 2,
        topk: int | None = None,
        ema_beta: float = 0.9,
    ):
        self.patience = int(patience)
        self.tol = float(tol)
        self.radius = float(radius)
        self.max_per_step = int(max_per_step)
        self.stuck_counts = torch.zeros(
            n_designs, dtype=torch.long
        )  # per-design counter
        self.ema = VectorEMA(size=n_designs, beta=ema_beta, bias_correction=True)
        if topk is None:
            topk = 0.5 * n_designs
        self.topk = max(int(topk), 1)

    def _ensure_device(self, device: torch.device):
        if self.stuck_counts.device != device:
            self.stuck_counts = self.stuck_counts.to(device)
        self.ema = self.ema.to(device)

    @torch.no_grad()
    def maybe_replace(
        self,
        designs: torch.Tensor,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
        per_design_loss: Optional[torch.Tensor],
        step: int,
    ) -> Dict[str, Any]:
        logs: Dict[str, Any] = {}
        if bounds is None or self.patience <= 0:
            return logs

        device = designs.device
        self._ensure_device(device)

        low, high = bounds
        low = low.to(device).expand_as(designs)
        high = high.to(device).expand_as(designs)

        # Detect boundary contact per design (any dim close to low or high)
        at_low = (designs - low).abs() <= self.tol
        at_high = (high - designs).abs() <= self.tol
        stuck_now = (at_low | at_high).view(designs.shape[0], -1).any(dim=1)

        # Update patience counters
        self.stuck_counts = torch.where(
            stuck_now, self.stuck_counts + 1, torch.zeros_like(self.stuck_counts)
        )
        # If not stuck now, reduce count by 1 (clamp at 0)
        not_stuck = ~stuck_now
        self.stuck_counts = torch.where(
            not_stuck,
            torch.clamp(self.stuck_counts - 1, min=0),
            self.stuck_counts,
        )

        # Candidates to reinit
        to_reinit = torch.nonzero(self.stuck_counts >= self.patience).flatten()
        if to_reinit.numel() == 0:
            logs["n_replaced"] = 0
            return logs

        # Need per-design loss to pick centers/exclusions
        if per_design_loss is None or per_design_loss.numel() != designs.shape[0]:
            logs["n_replaced"] = 0
            return logs

        # Update EMA loss
        self.ema.update(per_design_loss)
        ema_metric = self.ema.value()

        # Exclude topk from replacement
        topk_indices = torch.topk(ema_metric, k=self.topk, largest=False).indices
        to_reinit = to_reinit[~torch.isin(to_reinit, topk_indices)]
        if to_reinit.numel() == 0:
            logs["n_replaced"] = 0
            return logs

        # Limit number replaced per step
        if to_reinit.numel() > self.max_per_step:
            to_reinit = to_reinit[
                torch.randperm(to_reinit.numel(), device=device)[: self.max_per_step]
            ]

        x_new = propose_replacements_around_topk(
            designs=designs,
            bounds=(low, high),
            per_design_loss=ema_metric,
            to_reinit=to_reinit,
            topk=self.topk,
            radius=self.radius,
        )
        # Apply replacements and reset counters
        designs.data[to_reinit] = x_new
        self.stuck_counts[to_reinit] = 0
        self.ema.reset(to_reinit)

        logs["n_replaced"] = int(to_reinit.numel())
        logs["indices"] = to_reinit.detach().cpu()
        return logs


class BottomKReinitializer(ReplacementPolicy):
    """
    Reinitialize designs that have remained in the bottom-k (worst losses) for `patience` steps.
    Uses the same sampling logic as BoundaryReinitializer, proposing new points near random top-k centers.
    """

    def __init__(
        self,
        n_designs: int,
        patience: int = 50,
        max_per_step: int = 2,
        bottomk: int
        | None = None,  # how many worst designs to track; default: ceil(0.2 * n_designs)
        topk_centers: int
        | None = None,  # number of best designs to sample centers from
        radius: float = 0.05,  # normalized radius in [0,1] space
        ema_beta: float = 0.9,  # exponential moving average beta
        movement_threshold: float = 1e-3,  # threshold for "stuck" movement in [0,1] box
        movement_ema_beta: float = 0.9,  # EMA beta for movement tracking
    ):
        self.patience = int(patience)
        self.max_per_step = int(max_per_step)
        self.radius = float(radius)
        self.movement_threshold = float(movement_threshold)
        self.ema = VectorEMA(size=n_designs, beta=ema_beta, bias_correction=True)
        self.movement_ema = VectorEMA(
            size=n_designs, beta=movement_ema_beta, bias_correction=True
        )
        self.prev_designs = None  # cache previous positions

        if bottomk is None:
            bottomk = int(max(1, round(0.2 * n_designs)))
        if topk_centers is None:
            topk_centers = int(max(1, round(0.2 * n_designs)))

        self.bottomk = int(bottomk)
        self.topk_centers = int(topk_centers)

        # Track consecutive steps in bottom-k
        self.bottom_counts = torch.zeros(n_designs, dtype=torch.long)

    def _ensure_device(self, device: torch.device):
        if self.bottom_counts.device != device:
            self.bottom_counts = self.bottom_counts.to(device)
        self.ema = self.ema.to(device)
        self.movement_ema = self.movement_ema.to(device)
        if self.prev_designs is not None and self.prev_designs.device != device:
            self.prev_designs = self.prev_designs.to(device)

    @torch.no_grad()
    def maybe_replace(
        self,
        designs: torch.Tensor,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
        per_design_loss: Optional[torch.Tensor],
        step: int,
    ) -> Dict[str, Any]:
        logs: Dict[str, Any] = {}
        if bounds is None or self.patience <= 0:
            return logs
        if per_design_loss is None or per_design_loss.numel() != designs.shape[0]:
            return logs

        device = designs.device
        self._ensure_device(device)
        low, high = bounds
        low = low.to(device).expand_as(designs)
        high = high.to(device).expand_as(designs)

        # Compute per-design movement since last step in normalized [0,1] box
        z_current = _to_box_norm(designs, low, high)
        if self.prev_designs is None:
            # First step: initialize with zero movement
            movement = torch.zeros(designs.shape[0], device=device)
        else:
            z_prev = _to_box_norm(self.prev_designs, low, high)
            # L2 distance per design
            movement = (
                z_current.view(designs.shape[0], -1) - z_prev.view(designs.shape[0], -1)
            ).norm(dim=1)

        # Update movement EMA
        self.movement_ema.update(movement)
        avg_movement = self.movement_ema.value()

        # Cache current designs for next step
        self.prev_designs = designs.detach().clone()

        # Update EMA loss
        self.ema.update(per_design_loss)
        ema_metric = self.ema.value()

        # Identify current bottom-k (worst = largest loss)
        k = min(self.bottomk, ema_metric.numel())
        bottomk_indices = torch.topk(ema_metric, k=k, largest=True).indices
        in_bottom_now = torch.zeros(designs.shape[0], dtype=torch.bool, device=device)
        in_bottom_now[bottomk_indices] = True

        # Gate by top-k statistics (mean + std*factor)
        ks = min(self.topk_centers, ema_metric.numel())
        top_stats_indices = torch.topk(ema_metric, k=ks, largest=False).indices
        top_vals = ema_metric[top_stats_indices]
        top_mean = top_vals.mean()
        top_std = top_vals.std(unbiased=False)
        thr = top_mean + torch.clamp(top_std, min=1e-6)

        # Check if worse than threshold
        is_worse_enough = ema_metric > thr
        # Also check if stuck (low movement)
        is_stuck = avg_movement < self.movement_threshold
        # Combine conditions
        in_bottom_and_stuck = in_bottom_now & is_worse_enough & is_stuck

        # Update consecutive counters (only increment if BOTH bottom-k AND stuck)
        self.bottom_counts = torch.where(
            in_bottom_and_stuck,
            self.bottom_counts + 1,
            torch.zeros_like(self.bottom_counts),
        )
        # If not in bottom-k OR moving, reduce count by 1 (clamp at 0)
        not_problematic = ~in_bottom_and_stuck
        self.bottom_counts = torch.where(
            not_problematic,
            torch.clamp(self.bottom_counts - 1, min=0),
            self.bottom_counts,
        )
        # Candidates to reinit = those bottom AND stuck for >= patience
        to_reinit = torch.nonzero(self.bottom_counts >= self.patience).flatten()
        if to_reinit.numel() == 0:
            logs["n_replaced"] = 0
            logs["avg_movement"] = float(avg_movement.mean().item())
            return logs

        # Limit number replaced per step
        if to_reinit.numel() > self.max_per_step:
            to_reinit = to_reinit[
                torch.randperm(to_reinit.numel(), device=device)[: self.max_per_step]
            ]

        # Propose new positions around random members of the current top-k centers
        x_new = propose_replacements_around_topk(
            designs=designs,
            bounds=(low, high),
            per_design_loss=ema_metric,
            to_reinit=to_reinit,
            topk=self.topk_centers,
            radius=self.radius,
        )

        # Apply replacements and reset their counters
        designs.data[to_reinit] = x_new
        self.bottom_counts[to_reinit] = 0
        self.ema.reset(to_reinit)
        # CHANGED: reset movement EMA to mean (give benefit of the doubt for new positions)
        mean_movement = avg_movement.mean()
        if self.movement_ema.buffer[to_reinit].numel() > 0:
            self.movement_ema.buffer[to_reinit] = mean_movement

        # Update prev_designs after replacement
        self.prev_designs[to_reinit] = x_new

        logs["n_replaced"] = int(to_reinit.numel())
        logs["indices"] = to_reinit.detach().cpu()
        logs["avg_movement"] = float(avg_movement.mean().item())
        return logs


class IsStuckDetector:
    """
    Detects per-design stuck conditions:
      - boundary stuck: any dim near low/high for >= patience_boundary steps
      - low-movement stuck: movement EMA below movement_threshold for >= patience_movement steps
    """

    def __init__(
        self,
        n_designs: int,
        patience_boundary: int = 20,
        patience_movement: int = 50,
        movement_threshold: float = 1e-5,
        tol: float = 1e-6,
        movement_ema_beta: float = 0.9,
    ):
        self.patience_boundary = int(patience_boundary)
        self.patience_movement = int(patience_movement)
        self.movement_threshold = float(movement_threshold)
        self.tol = float(tol)

        self.boundary_counts = torch.zeros(n_designs, dtype=torch.long)
        self.movement_counts = torch.zeros(n_designs, dtype=torch.long)
        self.movement_ema = VectorEMA(
            size=n_designs, beta=movement_ema_beta, bias_correction=True
        )
        self.prev_designs = None
        self._n = int(n_designs)

    def to(self, device: torch.device):
        self.boundary_counts = self.boundary_counts.to(device)
        self.movement_counts = self.movement_counts.to(device)
        self.movement_ema = self.movement_ema.to(device)
        if self.prev_designs is not None:
            self.prev_designs = self.prev_designs.to(device)
        return self

    @torch.no_grad()
    def update(
        self, designs: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ):
        device = designs.device
        self.to(device)

        if bounds is None:
            # No box -> nothing to mark as boundary stuck
            stuck_boundary_now = torch.zeros(
                designs.shape[0], dtype=torch.bool, device=device
            )
            low = high = None
        else:
            low, high = bounds
            low = low.to(device).expand_as(designs)
            high = high.to(device).expand_as(designs)
            at_low = (designs - low).abs() <= self.tol
            at_high = (high - designs).abs() <= self.tol
            stuck_boundary_now = (
                (at_low | at_high).view(designs.shape[0], -1).any(dim=1)
            )

        # update boundary counters
        self.boundary_counts = torch.where(
            stuck_boundary_now,
            self.boundary_counts + 1,
            torch.clamp(self.boundary_counts - 1, min=0),
        )

        # movement (normalized to box if bounds provided)
        if bounds is None:
            # fallback: raw L2 movement
            if self.prev_designs is None:
                movement = torch.zeros(designs.shape[0], device=device)
            else:
                movement = (
                    designs.view(designs.shape[0], -1)
                    - self.prev_designs.view(designs.shape[0], -1)
                ).norm(dim=1)
        else:
            z_current = _to_box_norm(designs, low, high)
            if self.prev_designs is None:
                movement = torch.zeros(designs.shape[0], device=device)
            else:
                z_prev = _to_box_norm(self.prev_designs, low, high)
                movement = (
                    z_current.view(designs.shape[0], -1)
                    - z_prev.view(designs.shape[0], -1)
                ).norm(dim=1)

        self.movement_ema.update(movement)
        avg_movement = self.movement_ema.value()

        stuck_movement_now = avg_movement < self.movement_threshold
        self.movement_counts = torch.where(
            stuck_movement_now,
            self.movement_counts + 1,
            torch.clamp(self.movement_counts - 1, min=0),
        )

        self.prev_designs = designs.detach().clone()

        # If designs are not stuck now, reduce counts by 1 (clamp at 0)
        self.boundary_counts = decrease_counter(
            self.boundary_counts, ~stuck_boundary_now
        )
        self.movement_counts = decrease_counter(
            self.movement_counts, ~stuck_movement_now
        )

    def mask(self):
        """Return two boolean masks: (boundary_stuck_mask, movement_stuck_mask)"""
        boundary_mask = self.boundary_counts >= self.patience_boundary
        movement_mask = self.movement_counts >= self.patience_movement
        return boundary_mask | movement_mask

    def reset(self, idx: torch.Tensor):
        """
        Reset stuck counters and movement EMA for specified indices.
        If idx is None, reset all.
        """
        self.boundary_counts[idx] = 0
        self.movement_counts[idx] = 0
        self.movement_ema.reset(idx)


class CompositeReinitializer(ReplacementPolicy):
    """
    Compose an IsStuckDetector with bottom-k membership tracking.
    Reinitializes indices that have been stuck (boundary OR movement) in the bottom-k
    for >= patience_bottomk steps.
    """

    def __init__(
        self,
        n_designs: int,
        detector: IsStuckDetector,
        topk_centers: int | None = None,
        radius: float = 0.1,
        max_per_step: int = 2,
        bottomk: int | None = None,
        patience_bottomk: int = 50,
        ema_beta: float = 0.9,
    ):
        self.detector = detector
        self.radius = float(radius)
        self.max_per_step = int(max_per_step)
        if topk_centers is None:
            topk_centers = max(1, int(0.5 * n_designs))
        self.topk_centers = int(topk_centers)
        self.ema = VectorEMA(size=n_designs, beta=ema_beta, bias_correction=True)

        # Track bottom-k membership counts (how many steps a design has been stuck AND in bottom-k)
        if bottomk is None:
            bottomk = int(max(1, round(0.5 * n_designs)))
        self.bottomk = int(bottomk)
        self.patience_bottomk = int(patience_bottomk)
        self.bottom_counts = torch.zeros(n_designs, dtype=torch.long)

    @torch.no_grad()
    def maybe_replace(self, designs, bounds, per_design_loss, step: int):
        logs = {}
        if bounds is None or per_design_loss is None:
            logs["n_replaced"] = 0
            return logs
        device = designs.device
        self.detector.to(device)
        self.ema.to(device)

        # ensure bottom_counts on correct device
        if self.bottom_counts.device != device:
            self.bottom_counts = self.bottom_counts.to(device)

        # update detector state
        self.detector.update(designs, bounds)
        stuck_mask = self.detector.mask()

        # Update EMA loss
        self.ema.update(per_design_loss)
        ema_metric = self.ema.value()

        # Determine bottom-k membership using instantaneous per_design_loss
        k = min(self.bottomk, ema_metric.numel())
        bottomk_indices = torch.topk(ema_metric, k=k, largest=True).indices
        in_bottom_now = torch.zeros(designs.shape[0], dtype=torch.bool, device=device)
        in_bottom_now[bottomk_indices] = True

        # Increase bottom counters
        self.bottom_counts = torch.where(
            in_bottom_now,
            self.bottom_counts + 1,
            torch.clamp(self.bottom_counts - 1, min=0),
        )
        # If not stuck OR not in bottom-k, reduce count by 1 (clamp at 0)
        self.bottom_counts = decrease_counter(self.bottom_counts, ~in_bottom_now)

        # Candidates are those that have been stuck AND in bottom-k for >= patience_bottomk
        to_reinit = torch.nonzero(
            (self.bottom_counts >= self.patience_bottomk) & stuck_mask
        ).flatten()
        if to_reinit.numel() == 0:
            logs["n_replaced"] = 0
            return logs

        # limit replacements
        if to_reinit.numel() > self.max_per_step:
            to_reinit = to_reinit[
                torch.randperm(to_reinit.numel(), device=device)[: self.max_per_step]
            ]

        # propose and apply replacements
        low, high = bounds
        low = low.to(device).expand_as(designs)
        high = high.to(device).expand_as(designs)
        x_new = propose_replacements_around_topk(
            designs=designs,
            bounds=(low, high),
            per_design_loss=ema_metric,
            to_reinit=to_reinit,
            topk=self.topk_centers,
            radius=self.radius,
        )
        designs.data[to_reinit] = x_new

        # reset internal counters for replaced indices
        self.detector.reset(to_reinit)
        self.bottom_counts[to_reinit] = 0
        self.ema.reset(to_reinit)

        logs["n_replaced"] = int(to_reinit.numel())
        logs["indices"] = to_reinit.detach().cpu()
        return logs
