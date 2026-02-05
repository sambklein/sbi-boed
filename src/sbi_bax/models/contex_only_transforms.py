"""Implementations of transforms that are only a function of the input context."""

import numpy as np
import torch
from torch import nn

from nflows.transforms.base import Transform
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.splines.rational_quadratic import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)
from nflows.utils import torchutils

from sbi_bax.models.conditional_linear import ResidualNetwork


class ContextOnlyPiecewiseRationalQuadraticTransform(Transform):
    """
    Piecewise rational quadratic transform where parameters depend only on context,
    not on the input values (non-autoregressive).
    """

    def __init__(
        self,
        features,
        context_features,
        context_net: nn.Module,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.features = features
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        output_size = features * self._output_dim_multiplier()
        self.context_net = context_net(context_features, output_size)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError("tails must be 'linear' or None")

    def forward(self, inputs, context):
        """Forward transformation."""
        transform_params = self._get_transform_params(context)
        return self._apply_transform(inputs, transform_params, inverse=False)

    def inverse(self, inputs, context):
        """Inverse transformation."""
        transform_params = self._get_transform_params(context)
        return self._apply_transform(inputs, transform_params, inverse=True)

    def _get_transform_params(self, context):
        """Get transformation parameters from context."""
        batch_size = context.shape[0]

        # Get parameters from context network
        all_params = self.context_net(context)  # [batch_size, features * param_dim]

        # Reshape to [batch_size, features, param_dim]
        transform_params = all_params.view(
            batch_size, self.features, self._output_dim_multiplier()
        )

        return transform_params

    def _apply_transform(self, inputs, transform_params, inverse=False):
        """Apply the spline transformation."""
        # Extract spline parameters
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        # Optional: normalize by hidden features (as in original)
        if hasattr(self.context_net, "hidden_features"):
            hidden_features = self.context_net.hidden_features
        else:
            # Estimate from the network structure
            hidden_features = None
            for module in self.context_net:
                if isinstance(module, nn.Linear):
                    hidden_features = module.out_features
                    break

        if hidden_features is not None:
            unnormalized_widths /= np.sqrt(hidden_features)
            unnormalized_heights /= np.sqrt(hidden_features)

        # Choose spline function based on tails
        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError("tails must be 'linear' or None")

        # Apply spline transformation
        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)


def _legendre_nodes_weights(n: int):
    """Gauss–Legendre nodes/weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)  # nodes in [-1,1]
    u = (x + 1.0) / 2.0
    w = w / 2.0
    return u, w


def entropy_of_transform_1d(
    transform: ContextOnlyPiecewiseRationalQuadraticTransform,
    base_dist: torch.distributions.Distribution,
    context: torch.Tensor,
    n_points: int = 128,
) -> torch.Tensor:
    """
    Compute H(Y|context) for 1D transform Y = f(X; context).
    Uses: H(Y|c) = H(X) + E_X[log |f'(X; c)|], via Gauss–Legendre quadrature.

    Requirements:
      - transform.features == 1
      - base_dist.entropy() and base_dist.icdf(u) are available
      - context shape: [context_dim] or [batch, context_dim]
    Returns:
      Tensor scalar if single context, else [batch] with one entropy per context row.
    """
    if getattr(transform, "features", None) != 1:
        raise ValueError("entropy_of_transform_1d supports features == 1 only.")
    if not hasattr(base_dist, "icdf"):
        raise ValueError("base_dist must implement .icdf for quadrature.")
    base_H = base_dist.entropy()
    if base_H.ndim != 0:
        base_H = base_H.squeeze()

    # Prepare quadrature nodes in u-space and map via icdf
    u, w = _legendre_nodes_weights(n_points)
    device = context.device if context.is_cuda else torch.device("cpu")
    u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(w, dtype=torch.float32, device=device)  # [n_points]

    # Support batched contexts
    if context.ndim == 1:
        context = context.unsqueeze(0)  # [1, C]
    B, C = context.shape

    # x nodes shared across contexts (base does not depend on context)
    x = base_dist.icdf(u_t)  # [n_points]
    inputs = x.unsqueeze(-1)  # [n_points, 1]

    entropies = []
    for b in range(B):
        ctx_b = context[b].unsqueeze(0).expand(inputs.size(0), -1)  # [n_points, C]
        # forward returns log |dy/dx| per sample (since features==1)
        _, logabsdet = transform.forward(inputs, ctx_b)
        # Quadrature: E[log|f'|] = ∫_0^1 log|f'(F^{-1}(u))| du ≈ Σ w_i log|...|
        E_logdet = torch.dot(w_t, logabsdet)
        entropies.append(base_H.to(E_logdet) + E_logdet)

    entropies = torch.stack(entropies)
    return entropies.squeeze(0) if entropies.numel() == 1 else entropies


# Usage example:
def test_context_only_transform():
    """Test the context-only transform."""
    batch_size = 16
    features = 10
    context_features = 24

    # Create transform
    transform = ContextOnlyPiecewiseRationalQuadraticTransform(
        features=features,
        context_features=context_features,
        context_net=ResidualNetwork,
        num_bins=8,
        tails="linear",
    )

    # Test data
    inputs = torch.randn(batch_size, features)
    context = torch.randn(batch_size, context_features)

    # Forward pass
    outputs, logabsdet_forward = transform.forward(inputs, context)
    print(f"Forward - inputs: {inputs.shape}, outputs: {outputs.shape}")
    print(f"Forward logabsdet: {logabsdet_forward.shape}")

    # Inverse pass
    reconstructed, logabsdet_inverse = transform.inverse(outputs, context)
    print(f"Inverse - outputs: {outputs.shape}, reconstructed: {reconstructed.shape}")
    print(f"Inverse logabsdet: {logabsdet_inverse.shape}")

    # Check invertibility
    print(f"Reconstruction error: {torch.max(torch.abs(inputs - reconstructed))}")
    print(
        f"Logabsdet sum: {torch.max(torch.abs(logabsdet_forward + logabsdet_inverse))}"
    )


if __name__ == "__main__":
    test_context_only_transform()
