from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms.linear import Linear


class EmbeddingNet(nn.Module):
    """Base class for embedding network."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_final_layer(self):
        """Get the final layer of the network."""
        return self[-1]


class ResidualBlock(nn.Module):
    """A residual block with batch normalization and dropout."""

    def __init__(self, dim, activation=F.relu, dropout_prob=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_prob),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = activation

    @abstractmethod
    def __getitem__(self, index):
        """Make the network subscriptable to access layers."""
        pass

    def forward(self, x):
        return self.activation(x + self.layers(x))


class ResidualNetwork(EmbeddingNet):
    """Residual network for parameter prediction."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        num_blocks=2,
        activation=F.relu,
        dropout_prob=0.0,
    ):
        super().__init__(input_dim, output_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, activation, dropout_prob)
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

        # Create a list of all layers for indexing
        self._layers = (
            [self.input_layer] + list(self.residual_blocks) + [self.output_layer]
        )

    def __getitem__(self, index):
        """Make the network subscriptable."""
        return self._layers[index]

    def __len__(self):
        """Return the number of layers."""
        return len(self._layers)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class MLPNetwork(EmbeddingNet):
    """Standard MLP network with optional features."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        num_blocks=2,
        activation=F.relu,
        dropout_prob=0.0,
    ):
        super().__init__(input_dim, output_dim)

        layers = []
        current_dim = input_dim

        # Input layer
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.Dropout(dropout_prob)])
        current_dim = hidden_dim

        # Hidden layers
        for i in range(num_blocks - 1):
            layers.extend(
                [nn.Linear(current_dim, hidden_dim), nn.Dropout(dropout_prob)]
            )

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.activation = activation

    def __getitem__(self, index):
        """Make the network subscriptable."""
        return self.network[index]

    def forward(self, x):
        # Apply activation to all layers except the last
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear) and i < len(self.network) - 1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x


class ConditionalLuLinear(Linear):
    """A conditional linear transform where LU decomposition parameters are predicted from context."""

    def __init__(
        self,
        features: torch.Tensor,
        context_dim: torch.Tensor,
        embedding_net: EmbeddingNet = MLPNetwork,
        using_cache=False,
        identity_init=False,
        eps=1e-3,
    ):
        super().__init__(features, using_cache)

        self.eps = eps

        # LU matrix indices
        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        # Number of parameters to predict
        n_triangular_entries = ((features - 1) * features) // 2
        self.n_lower_entries = n_triangular_entries
        self.n_upper_entries = n_triangular_entries
        self.n_diag_entries = features
        self.n_bias_entries = features

        total_params = (
            self.n_lower_entries
            + self.n_upper_entries
            + self.n_diag_entries
            + self.n_bias_entries
        )

        self.parameter_net = embedding_net(context_dim, total_params)

        # Initialize the network for identity transform
        if identity_init:
            self._initialize_for_identity()

    def _initialize_for_identity(self):
        """Initialize the network to produce identity transformation."""
        with torch.no_grad():
            final_layer = self.parameter_net.get_final_layer()

            if hasattr(final_layer, "bias") and final_layer.bias is not None:
                bias = final_layer.bias

                # Lower triangular entries (zeros for identity)
                bias[: self.n_lower_entries] = 0.0

                # Upper triangular entries (zeros for identity)
                start_idx = self.n_lower_entries
                end_idx = start_idx + self.n_upper_entries
                bias[start_idx:end_idx] = 0.0

                # Upper diagonal entries (log(softplus^{-1}(1 - eps)))
                start_idx = end_idx
                end_idx = start_idx + self.n_diag_entries
                constant = np.log(np.exp(1 - self.eps) - 1)
                bias[start_idx:end_idx] = constant

                # Bias entries (zeros)
                bias[end_idx:] = 0.0

            # Zero out the final layer weights so output = bias
            if hasattr(final_layer, "weight") and final_layer.weight is not None:
                final_layer.weight.zero_()

    def _extract_parameters(self, params):
        """Extract LU decomposition parameters from network output."""
        start_idx = 0

        # Lower triangular entries
        lower_entries = params[..., start_idx : start_idx + self.n_lower_entries]
        start_idx += self.n_lower_entries

        # Upper triangular entries
        upper_entries = params[..., start_idx : start_idx + self.n_upper_entries]
        start_idx += self.n_upper_entries

        # Upper diagonal entries (unconstrained)
        unconstrained_upper_diag = params[
            ..., start_idx : start_idx + self.n_diag_entries
        ]
        start_idx += self.n_diag_entries

        # Bias
        bias = params[..., start_idx:]

        return lower_entries, upper_entries, unconstrained_upper_diag, bias

    def _create_lower_upper(self, context):
        """Create lower and upper matrices from context."""
        batch_size = context.shape[0]
        device = context.device

        # Predict parameters from context
        params = self.parameter_net(context)
        lower_entries, upper_entries, unconstrained_upper_diag, bias = (
            self._extract_parameters(params)
        )

        # Upper diagonal with constraints
        upper_diag = F.softplus(unconstrained_upper_diag) + self.eps

        # Create batch of lower matrices
        lower = torch.zeros(batch_size, self.features, self.features, device=device)
        lower[:, self.lower_indices[0], self.lower_indices[1]] = lower_entries
        lower[:, self.diag_indices[0], self.diag_indices[1]] = 1.0  # Unit diagonal

        # Create batch of upper matrices
        upper = torch.zeros(batch_size, self.features, self.features, device=device)
        upper[:, self.upper_indices[0], self.upper_indices[1]] = upper_entries
        upper[:, self.diag_indices[0], self.diag_indices[1]] = upper_diag

        return lower, upper, bias, upper_diag

    def forward_no_cache(self, inputs, context):
        """Forward pass with context conditioning."""
        batch_size = inputs.shape[0]

        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1)

        lower, upper, bias, upper_diag = self._create_lower_upper(context)

        # Batch matrix multiplication
        outputs = torch.bmm(inputs.unsqueeze(1), upper.transpose(-2, -1)).squeeze(1)
        outputs = torch.bmm(outputs.unsqueeze(1), lower.transpose(-2, -1)).squeeze(1)
        outputs = outputs + bias

        # Log absolute determinant
        logabsdet = torch.sum(torch.log(upper_diag), dim=-1)

        return outputs, logabsdet

    def inverse_no_cache(self, inputs, context):
        """Inverse pass with context conditioning."""
        batch_size = inputs.shape[0]

        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1)

        lower, upper, bias, upper_diag = self._create_lower_upper(context)

        # Subtract bias
        outputs = inputs - bias

        # Solve triangular systems in batch
        outputs = outputs.unsqueeze(-1)

        # Solve Ly = (x - b)
        outputs = torch.triangular_solve(
            outputs, lower, upper=False, unitriangular=True
        )[0]

        # Solve Uz = y
        outputs = torch.triangular_solve(
            outputs, upper, upper=True, unitriangular=False
        )[0]

        outputs = outputs.squeeze(-1)

        # Log absolute determinant (negative for inverse)
        logabsdet = -torch.sum(torch.log(upper_diag), dim=-1)

        return outputs, logabsdet

    def weight(self, context):
        """Return the weight matrix."""
        lower, upper, _, _ = self._create_lower_upper(context)
        return torch.bmm(lower, upper)

    def weight_inverse(self, context):
        """Return the inverse weight matrix."""
        lower, upper, _, _ = self._create_lower_upper(context)
        identity = torch.eye(self.features, device=context.device)
        lower_inverse, _ = torch.triangular_solve(
            identity, lower, upper=False, unitriangular=True
        )
        return torch.triangular_solve(
            lower_inverse, upper, upper=True, unitriangular=False
        )[0]

    def logabsdet(self, context):
        _, _, _, upper_diag = self._create_lower_upper(context)
        return torch.sum(torch.log(upper_diag), dim=-1)

    def set_cache(self, context):
        """Set the cache for the transform."""
        self.cache.weight = self.weight(context)
        self.cache.inverse = self.weight_inverse(context)
        self.cache.logabsdet = self.logabsdet(context)

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            return super().forward(inputs, context)
        else:
            return self.forward_no_cache(inputs, context)

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            return super().inverse(inputs, context)
        else:
            return self.inverse_no_cache(inputs, context)


class ConditionalDiagonalLinear(Linear):
    """A conditional linear transform y = A * x + b where A is diagonal and predicted from context."""

    def __init__(
        self,
        features: int,
        context_dim: int,
        embedding_net: EmbeddingNet = MLPNetwork,
        using_cache=False,
        identity_init=False,
        eps=1e-3,
    ):
        super().__init__(features, using_cache)

        self.eps = eps
        self.features = features

        # Number of parameters to predict: diagonal + bias
        self.n_diag_entries = features
        self.n_bias_entries = features
        total_params = self.n_diag_entries + self.n_bias_entries

        self.parameter_net = embedding_net(context_dim, total_params)

        # Initialize the network for identity transform
        if identity_init:
            self._initialize_for_identity()

    def _initialize_for_identity(self):
        """Initialize the network to produce identity transformation (diag=1, bias=0)."""
        with torch.no_grad():
            final_layer = self.parameter_net.get_final_layer()

            if hasattr(final_layer, "bias") and final_layer.bias is not None:
                bias = final_layer.bias

                # Diagonal entries: log(softplus^{-1}(1 - eps)) so softplus gives ~1
                constant = np.log(np.exp(1 - self.eps) - 1)
                bias[: self.n_diag_entries] = constant

                # Bias entries (zeros for identity)
                bias[self.n_diag_entries :] = 0.0

            # Zero out the final layer weights so output = bias
            if hasattr(final_layer, "weight") and final_layer.weight is not None:
                final_layer.weight.zero_()

    def _extract_parameters(self, params):
        """Extract diagonal and bias parameters from network output."""
        # Diagonal entries (unconstrained)
        unconstrained_diag = params[..., : self.n_diag_entries]

        # Bias
        bias = params[..., self.n_diag_entries :]

        return unconstrained_diag, bias

    def _create_diagonal_bias(self, context):
        """Create diagonal scaling and bias from context."""
        # Predict parameters from context
        params = self.parameter_net(context)
        unconstrained_diag, bias = self._extract_parameters(params)

        # Diagonal with positive constraint
        diag = F.softplus(unconstrained_diag) + self.eps

        return diag, bias

    def forward_no_cache(self, inputs, context):
        """Forward pass: y = diag(A) * x + b"""
        batch_size = inputs.shape[0]

        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1)

        diag, bias = self._create_diagonal_bias(context)

        # Element-wise multiplication with diagonal
        outputs = inputs * diag + bias

        # Log absolute determinant (sum of log diagonal entries)
        logabsdet = torch.sum(torch.log(diag), dim=-1)

        return outputs, logabsdet

    def inverse_no_cache(self, inputs, context):
        """Inverse pass: x = (y - b) / diag(A)"""
        batch_size = inputs.shape[0]

        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1)

        diag, bias = self._create_diagonal_bias(context)

        # Inverse: (y - b) / diag
        outputs = (inputs - bias) / diag

        # Log absolute determinant (negative for inverse)
        logabsdet = -torch.sum(torch.log(diag), dim=-1)

        return outputs, logabsdet

    def weight(self, context):
        """Return the diagonal weight matrix as a full matrix."""
        diag, _ = self._create_diagonal_bias(context)
        batch_size = diag.shape[0]
        weight_matrix = torch.zeros(
            batch_size, self.features, self.features, device=diag.device
        )
        # Fill diagonal
        for i in range(batch_size):
            weight_matrix[i].fill_diagonal_(diag[i])
        return weight_matrix

    def weight_inverse(self, context):
        """Return the inverse diagonal weight matrix."""
        diag, _ = self._create_diagonal_bias(context)
        batch_size = diag.shape[0]
        inv_weight_matrix = torch.zeros(
            batch_size, self.features, self.features, device=diag.device
        )
        # Fill diagonal with 1/diag
        for i in range(batch_size):
            inv_weight_matrix[i].fill_diagonal_(1.0 / diag[i])
        return inv_weight_matrix

    def logabsdet(self, context):
        """Return log absolute determinant."""
        diag, _ = self._create_diagonal_bias(context)
        return torch.sum(torch.log(diag), dim=-1)

    def set_cache(self, context):
        """Set the cache for the transform."""
        diag, bias = self._create_diagonal_bias(context)
        self.cache.diag = diag
        self.cache.bias = bias
        self.cache.logabsdet = torch.sum(torch.log(diag), dim=-1)

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            # Use cached diagonal and bias
            outputs = inputs * self.cache.diag + self.cache.bias
            logabsdet = self.cache.logabsdet.expand(inputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs, context)

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            # Use cached diagonal and bias
            outputs = (inputs - self.cache.bias) / self.cache.diag
            logabsdet = -self.cache.logabsdet.expand(inputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs, context)
