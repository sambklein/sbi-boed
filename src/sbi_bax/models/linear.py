# A simple generic MLP model for encoding data.
import torch.nn as nn
import torch


class LinearEncoder(nn.Linear):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super(LinearEncoder, self).__init__(input_dim, output_dim)
        # Register a buffer for the sample means
        self.register_buffer("s_min", torch.zeros(input_dim))
        # Register a buffer for the sample stds
        self.register_buffer("s_denom", torch.ones(input_dim))

    def fit_norm(self, data):
        """Fit normalization parameters based on data."""
        # Assuming data is a tensor of shape (batch_size, input_dim)
        self.s_min = data.min(dim=0)[0]
        self.s_denom = data.max(dim=0)[0] - self.s_min
        # Avoid division by zero
        self.s_denom[self.s_denom == 0] = 1.0

    def forward(self, x):
        # Normalize the input using the fitted mean and std
        x = (x - self.s_min) / self.s_denom
        # Forward pass through the MLP
        return super().forward(x)
