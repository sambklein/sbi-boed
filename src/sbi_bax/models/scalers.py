import torch as T
from torch import nn
import torch
from sbi_bax.models.modules import IterativeNormLayer


class SimpleScaler(nn.Module):
    def __init__(self, low: list | float, high: list | float):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))
        self.register_buffer("eps", torch.as_tensor(1e-8, dtype=torch.float32))

    def forward(self, data, last_only=False):
        return (data - self.low) / (self.high - self.low + self.eps)

    def inverse(self, data):
        return data * (self.high - self.low + self.eps) + self.low


class ScaleToRange(nn.Module):
    """Affine scaling from [inp_low, inp_high] → [out_low, out_high]."""

    def __init__(self, inp_low, inp_high, out_low=-5.0, out_high=5.0):
        super().__init__()
        self.register_buffer("inp_low", torch.as_tensor(inp_low, dtype=torch.float32))
        self.register_buffer("inp_high", torch.as_tensor(inp_high, dtype=torch.float32))
        self.register_buffer("out_low", torch.as_tensor(out_low, dtype=torch.float32))
        self.register_buffer("out_high", torch.as_tensor(out_high, dtype=torch.float32))

    def forward(self, data: torch.Tensor, last_only=False, n_obs=None) -> torch.Tensor:
        # Normalize to [0, 1]
        normed = (data - self.inp_low) / (self.inp_high - self.inp_low + 1e-8)
        # Map to [out_low, out_high]
        return normed * (self.out_high - self.out_low) + self.out_low

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # Map back to [0, 1]
        normed = (data - self.out_low) / (self.out_high - self.out_low + 1e-8)
        # Map to [inp_low, inp_high]
        return normed * (self.inp_high - self.inp_low) + self.inp_low


class FlattenNormLayer(IterativeNormLayer):
    """Normalisation layer that flattens all but the last dimension before collecting stats."""

    def _flatten(self, inpt: T.Tensor) -> tuple[T.Tensor, tuple[int, ...]]:
        """Flatten all but the last dimension and return original shape."""
        orig_shape = inpt.shape
        flat = inpt.view(-1, orig_shape[-1])
        return flat, orig_shape

    def forward(
        self,
        inpt: T.Tensor,
        mask: T.BoolTensor | None = None,
        last_only: bool = False,
        n_obs: int = 1,
    ) -> T.Tensor:
        """Apply standardisation, collecting stats over flattened dimensions."""
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_forward and grad_setting)

        sel_inpt = self._mask(inpt, mask)
        sel_inpt_flat, orig_shape = self._flatten(sel_inpt)

        if self.training and not self.frozen and n_obs == 1:
            with T.no_grad():
                if last_only:
                    super().update(inpt[:, -1])
                else:
                    super().update(sel_inpt_flat)

        normed_flat = (sel_inpt_flat - self.means) / (self.vars.sqrt() + 1e-8)
        normed_inpt = normed_flat.view(orig_shape)

        normed_inpt = self._unmask(inpt, normed_inpt, mask)
        T.set_grad_enabled(grad_setting)
        return normed_inpt

    def inverse(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        """Undo the normalisation (alias for reverse)."""
        # just reuse the parent method to avoid duplication
        return super().reverse(inpt, mask)
