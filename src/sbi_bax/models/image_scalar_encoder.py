# Models for crystal structure prediction using a CNN-MLP encoder and a Transformer architecture.
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class Cnn_Mlp_Encoder(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (250, 250),
        scalar_dim: int = 3,
        output_dim: int = 32,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.image_pixels = np.prod(image_shape)
        self.scalar_dim = scalar_dim
        self.output_dim = output_dim

        # Downsample early as image is sparse
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        im_odim = 32  # Output dimension after CNN

        # # Mobile Net-like architecture for image encoding
        # self.image_encoder = nn.Sequential(
        #     # Stem
        #     nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.SiLU(),  # Swish activation (more expressive than ReLU)
        #     # Depthwise separable convolutions (more efficient)
        #     nn.Conv2d(
        #         32, 32, kernel_size=3, stride=1, padding=1, groups=32
        #     ),  # Depthwise
        #     nn.Conv2d(32, 64, kernel_size=1),  # Pointwise
        #     nn.BatchNorm2d(64),
        #     nn.SiLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
        #     nn.Conv2d(64, 128, kernel_size=1),
        #     nn.BatchNorm2d(128),
        #     nn.SiLU(),
        #     # Global context
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )
        # im_odim = 128  # Output dimension after CNN

        # Fully connected layers for scalar encoding
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # Combined fully connected layers
        self.combined = nn.Sequential(
            nn.Linear(im_odim + 32, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.output_dim),
        )
        # Register a buffer for the sample means
        self.register_buffer("s_min", torch.zeros(self.scalar_dim))
        # Register a buffer for the sample stds
        self.register_buffer("s_denom", torch.ones(self.scalar_dim))

    def split_image_scalar(self, x_flat):
        """
        Split the flattened input into image and scalar parts.
        """
        # Flatten the input to separate image and scalar parts
        x_flat = x_flat.view(-1, self.image_pixels + self.scalar_dim)
        # Split the flattened input into image and scalar parts
        image = x_flat[:, : self.image_pixels].reshape(-1, 1, *self.image_shape)
        scalars = x_flat[:, self.image_pixels :]
        return image, scalars

    def fit_norm(self, data):
        """
        Fit the scalar normalization layer to the data.
        """
        # Assuming data is a tensor of shape (batch_size, n_samples, ...)
        _, scalars = self.split_image_scalar(data)
        # Extract the max and min of the scalars (will be many repeats in scalars so mean is skewed)
        self.s_min = scalars.min(dim=0)[0]
        self.s_denom = scalars.max(dim=0)[0] - self.s_min
        # Avoid division by zero
        self.s_denom[self.s_denom == 0] = 1.0

    def scalar_norm(self, scalars):
        """
        Normalize the scalar part of the input.
        """
        # Normalize the scalars using the fitted mean and std
        return (scalars - self.s_min) / self.s_denom

    def forward(self, x_flat):
        batch_size, n_samples = x_flat.shape[:2]
        # Flatten the input to separate image and scalar parts
        image, scalars = self.split_image_scalar(x_flat)
        # Normalize scalars if the model has been called before
        scalars = self.scalar_norm(scalars)
        # Encode image and scalars
        image_feat = self.image_encoder(image)
        scalar_feat = self.scalar_encoder(scalars)
        combined = self.combined(torch.cat([image_feat, scalar_feat], dim=-1))
        return combined.view(batch_size, n_samples, -1)
