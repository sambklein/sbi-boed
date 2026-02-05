import torch
from nflows.transforms.base import Transform


class BatchedCartesianToHypersphericalTransform(Transform):
    """
    Transform multiple N-D Cartesian vectors to hyperspherical coordinates.

    Example:
        Input: [x1_1, x1_2, x1_3, x2_1, x2_2, x2_3]  (6 features = 2 vectors of dim 3)
        Output: [r1, θ1, φ1, r2, θ2, φ2]

    Each vector is independently transformed to hyperspherical coordinates:
        2D: (x, y) → (r, θ)
        3D: (x, y, z) → (r, θ, φ)
        ND: (x₁, ..., xₙ) → (r, θ₁, ..., θₙ₋₁)
    """

    def __init__(self, n_features, n_vectors, eps=1e-6):
        """
        Args:
            n_features: Total number of features
            n_vectors: Number of vectors to transform
            eps: Small constant for numerical stability
        """
        super().__init__()
        if n_features % n_vectors != 0:
            raise ValueError(
                f"n_features must be divisible by n_vectors, "
                f"got {n_features} and {n_vectors}"
            )

        self.n_features = n_features
        self.n_vectors = n_vectors
        self.dim_per_vector = n_features // n_vectors
        self.eps = eps

        if self.dim_per_vector < 2:
            raise ValueError(
                f"Each vector must have at least 2 dimensions, "
                f"got {self.dim_per_vector}"
            )

    def forward(self, inputs, context=None):
        """
        Transform from Cartesian to hyperspherical coordinates.

        Args:
            inputs: Tensor of shape [..., n_features]

        Returns:
            outputs: Tensor of shape [..., n_features] with hyperspherical coords
            logabsdet: Log absolute determinant of Jacobian
        """
        batch_shape = inputs.shape[:-1]

        # Reshape to [..., n_vectors, dim_per_vector]
        vectors = inputs.reshape(*batch_shape, self.n_vectors, self.dim_per_vector)

        outputs_list = []
        logabsdet_total = torch.zeros(batch_shape, device=inputs.device)

        # Transform each vector independently
        for i in range(self.n_vectors):
            vector = vectors[..., i, :]  # [..., dim_per_vector]

            # Compute radius
            r = torch.sqrt((vector**2).sum(dim=-1, keepdim=True) + self.eps)

            # Compute angles
            angles = []
            for j in range(self.dim_per_vector - 1):
                if j == self.dim_per_vector - 2:
                    # Last angle: use atan2 for full [-π, π] range
                    angle = torch.atan2(vector[..., -1], vector[..., -2])
                else:
                    # Other angles: use arccos for [0, π] range
                    numerator = vector[..., j]
                    denominator = torch.sqrt(
                        (vector[..., j:] ** 2).sum(dim=-1) + self.eps
                    )
                    cos_angle = torch.clamp(
                        numerator / denominator, -1 + self.eps, 1 - self.eps
                    )
                    angle = torch.arccos(cos_angle)

                angles.append(angle)

            # Stack: [r, θ₁, ..., θₙ₋₁]
            output = torch.cat([r] + [a.unsqueeze(-1) for a in angles], dim=-1)
            outputs_list.append(output)

            # Compute log absolute determinant for this vector
            logdet = (self.dim_per_vector - 1) * torch.log(r.squeeze(-1))
            for k, angle in enumerate(angles[:-1]):
                power = self.dim_per_vector - 2 - k
                logdet += power * torch.log(torch.sin(angle) + self.eps)

            logabsdet_total += logdet

        # Stack and reshape back to [..., n_features]
        outputs = torch.stack(outputs_list, dim=-2).reshape(
            *batch_shape, self.n_features
        )

        return outputs, logabsdet_total

    def inverse(self, inputs, context=None):
        """
        Transform from hyperspherical to Cartesian coordinates.

        Args:
            inputs: Tensor of shape [..., n_features] with hyperspherical coords

        Returns:
            outputs: Tensor of shape [..., n_features] with Cartesian coords
            logabsdet: Log absolute determinant of Jacobian
        """
        batch_shape = inputs.shape[:-1]

        # Reshape to [..., n_vectors, dim_per_vector]
        vectors = inputs.reshape(*batch_shape, self.n_vectors, self.dim_per_vector)

        outputs_list = []
        logabsdet_total = torch.zeros(batch_shape, device=inputs.device)

        # Transform each vector independently
        for i in range(self.n_vectors):
            vector = vectors[..., i, :]  # [..., dim_per_vector]

            r = vector[..., 0]
            angles = [vector[..., j] for j in range(1, self.dim_per_vector)]

            # Build Cartesian coordinates
            coords = []
            cumulative_sin = torch.ones_like(r)

            for j in range(self.dim_per_vector):
                if j < self.dim_per_vector - 1:
                    coord = r * cumulative_sin * torch.cos(angles[j])
                    cumulative_sin = cumulative_sin * torch.sin(angles[j])
                else:
                    coord = r * cumulative_sin

                coords.append(coord)

            output = torch.stack(coords, dim=-1)
            outputs_list.append(output)

            # Compute log absolute determinant for this vector
            logdet = -(self.dim_per_vector - 1) * torch.log(r + self.eps)
            for k, angle in enumerate(angles[:-1]):
                power = self.dim_per_vector - 2 - k
                logdet -= power * torch.log(torch.sin(angle) + self.eps)

            logabsdet_total += logdet

        # Stack and reshape back to [..., n_features]
        outputs = torch.stack(outputs_list, dim=-2).reshape(
            *batch_shape, self.n_features
        )

        return outputs, logabsdet_total
