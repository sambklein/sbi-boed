import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
        zero_init_output: bool = False,
        flatten_input: bool = False,
        flatten_output: bool = False,
        **kwargs,
    ):
        super().__init__()

        layers = [nn.Flatten()] if flatten_input else []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.SiLU())
            prev_dim = dim

        # Final layer
        final_layer = nn.Linear(prev_dim, output_dim)
        layers.append(final_layer)
        if flatten_output:
            layers.append(nn.Flatten())
        self.network = nn.Sequential(*layers)

        # Optional zero-out initialization for output layer
        if zero_init_output:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        # Register buffers for normalization
        self.register_buffer("s_min", torch.zeros(input_dim))
        self.register_buffer("s_denom", torch.ones(input_dim))

    def fit_norm(self, data):
        """Fit normalization parameters based on data."""
        self.s_min = data.min(dim=0)[0]
        self.s_denom = data.max(dim=0)[0] - self.s_min
        self.s_denom[self.s_denom == 0] = 1.0

    def forward(self, x):
        x = (x - self.s_min) / self.s_denom
        return self.network(x)
