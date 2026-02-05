import torch
import torch.nn as nn
from sbi_bax.models.mlp import Mlp


class ConditionalNCE(nn.Module):
    """
    Conditional InfoNCE estimator for I(X; Y | Z=z).

    Estimates I(X; Y | Z) using the InfoNCE bound:
        I(X; Y | Z) ≥ E_{x,y,z}[log(exp(T(x,y,z)) / sum_j exp(T(x,y_j,z)))]

    Input shapes:
        x: (n_z, n_samples, x_dim) - query variable
        y: (n_z, n_samples, y_dim) - target variable
        z: (n_z, n_samples, z_dim) - conditioning variable

    Returns:
        MI estimate per z: (n_z,)
    """

    def __init__(self, x_dim, y_dim, z_dim, hidden=(256, 256)):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.in_dim = x_dim + y_dim + z_dim

        # Critic network T(x, y, z) → scalar score
        self.critic = Mlp(self.in_dim, 1, hidden_dims=hidden)

    def train_critic(self, x, y, z):
        # Flatten to 2D
        x_flat = x.reshape(-1, self.x_dim)
        y_flat = y.reshape(-1, self.y_dim)
        z_flat = z.reshape(-1, self.z_dim)
        # Positive batch
        pos_inp = torch.cat([x_flat, y_flat, z_flat], dim=-1)
        pos_scores = self.critic(pos_inp)

        # Negative batch (shuffled y)
        y_neg = y_flat[torch.randperm(y_flat.shape[0])]
        neg_inp = torch.cat([x_flat, y_neg, z_flat], dim=-1)
        neg_scores = self.critic(neg_inp)

        # Binary cross-entropy loss
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = nn.BCEWithLogitsLoss()(scores, labels).view(-1)
        return -loss

    def forward_T(self, x, y, z):
        """
        Compute critic scores T(x, y, z).

        Args:
            x: (batch, x_dim) or (n_z, n_samples, x_dim)
            y: (batch, y_dim) or (n_z, n_samples, y_dim)
            z: (batch, z_dim) or (n_z, n_samples, z_dim)

        Returns:
            T: same batch shape as inputs
        """
        original_shape = x.shape[:-1]

        # Flatten to 2D
        x_flat = x.reshape(-1, self.x_dim)
        y_flat = y.reshape(-1, self.y_dim)
        z_flat = z.reshape(-1, self.z_dim)

        # Concatenate and score
        inp = torch.cat([x_flat, y_flat, z_flat], dim=1)
        t = self.critic(inp)

        # Reshape back
        return t.view(*original_shape)

    def fit_norm(self, x, y, z):
        """Fit normalization for critic based on data."""
        inp = torch.cat(
            [
                x.reshape(-1, self.x_dim),
                y.reshape(-1, self.y_dim),
                z.reshape(-1, self.z_dim),
            ],
            dim=1,
        )
        self.critic.fit_norm(inp)  # type: ignore

    def mi_estimate(self, x, y, z):
        """
        Estimate I(X; Y | Z=z) for each z using InfoNCE.

        For each conditioning value z_i:
        - Positive pair: (x_i, y_i, z_i)
        - Negative pairs: (x_i, y_j, z_i) for j ≠ i

        Args:
            x: (n_z, n_samples, x_dim)
            y: (n_z, n_samples, y_dim)
            z: (n_z, n_samples, z_dim)

        Returns:
            MI per z: (n_z,)
        """
        n_z, n_samples, _ = x.shape

        # ===== Compute pairwise scores T(x_i, y_j, z_i) =====
        # Expand dimensions for broadcasting
        # x_i: (n_z, n_samples, 1, x_dim) - each x paired with all y
        # y_j: (n_z, 1, n_samples, y_dim) - all y paired with each x
        # z_i: (n_z, n_samples, 1, z_dim) - same z for all pairs with same x_i

        x_expanded = x.unsqueeze(2)  # (n_z, n_samples, 1, x_dim)
        y_expanded = y.unsqueeze(1)  # (n_z, 1, n_samples, y_dim)
        z_expanded = z.unsqueeze(2)  # (n_z, n_samples, 1, z_dim)

        # Broadcast to (n_z, n_samples, n_samples, dim)
        x_pairs = x_expanded.expand(-1, -1, n_samples, -1)
        y_pairs = y_expanded.expand(-1, n_samples, -1, -1)
        z_pairs = z_expanded.expand(-1, -1, n_samples, -1)

        # Concatenate and flatten for critic
        pairs = torch.cat([x_pairs, y_pairs, z_pairs], dim=-1)
        pairs_flat = pairs.reshape(-1, self.in_dim)

        # Compute all pairwise scores
        t_pairs_flat = self.critic(pairs_flat)
        t_pairs = t_pairs_flat.view(n_z, n_samples, n_samples)

        # ===== InfoNCE estimator =====
        # Positive scores: diagonal T(x_i, y_i, z_i)
        log_pos = torch.diagonal(t_pairs, dim1=1, dim2=2)  # (n_z, n_samples)

        # Denominator: sum over all negatives (including positive)
        # log sum_j exp(T(x_i, y_j, z_i))
        log_denom = torch.logsumexp(t_pairs, dim=2)  # (n_z, n_samples)

        # InfoNCE per sample: log(exp(T_pos) / sum_j exp(T_j))
        log_ratio = log_pos - log_denom  # (n_z, n_samples)

        # Average over samples per z
        mi_per_z = log_ratio.mean(dim=1)  # (n_z,)

        return mi_per_z


class ConditionalNCEEfficient(nn.Module):
    """
    Memory-efficient version using chunked computation.
    """

    def __init__(self, x_dim, y_dim, z_dim, hidden=(256, 256), chunk_size=128):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.in_dim = x_dim + y_dim + z_dim
        self.chunk_size = chunk_size

        self.critic = Mlp(self.in_dim, 1, hidden_dims=hidden)

    def forward_T(self, x, y, z):
        """Compute critic scores."""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.x_dim)
        y_flat = y.reshape(-1, self.y_dim)
        z_flat = z.reshape(-1, self.z_dim)
        inp = torch.cat([x_flat, y_flat, z_flat], dim=1)
        t = self.critic(inp)
        return t.view(*original_shape)

    def mi_estimate(self, x, y, z):
        """
        Memory-efficient MI estimation using chunked computation.

        Args:
            x: (n_z, n_samples, x_dim)
            y: (n_z, n_samples, y_dim)
            z: (n_z, n_samples, z_dim)

        Returns:
            MI per z: (n_z,)
        """
        n_z, n_samples, _ = x.shape
        device = x.device

        # Storage for log ratios
        log_ratios = torch.zeros(n_z, n_samples, device=device)

        # Process in chunks to save memory
        for i in range(0, n_samples, self.chunk_size):
            end_i = min(i + self.chunk_size, n_samples)
            chunk_size_i = end_i - i

            # Get chunk of x's and z's
            x_chunk = x[:, i:end_i, :]  # (n_z, chunk_size_i, x_dim)
            z_chunk = z[:, i:end_i, :]  # (n_z, chunk_size_i, z_dim)

            # Expand for pairing with all y's
            x_expanded = x_chunk.unsqueeze(2)  # (n_z, chunk_size_i, 1, x_dim)
            y_expanded = y.unsqueeze(1)  # (n_z, 1, n_samples, y_dim)
            z_expanded = z_chunk.unsqueeze(2)  # (n_z, chunk_size_i, 1, z_dim)

            # Broadcast
            x_pairs = x_expanded.expand(-1, -1, n_samples, -1)
            y_pairs = y_expanded.expand(-1, chunk_size_i, -1, -1)
            z_pairs = z_expanded.expand(-1, -1, n_samples, -1)

            # Compute scores for this chunk
            pairs = torch.cat([x_pairs, y_pairs, z_pairs], dim=-1)
            pairs_flat = pairs.reshape(-1, self.in_dim)
            t_pairs_flat = self.critic(pairs_flat)
            t_pairs = t_pairs_flat.view(n_z, chunk_size_i, n_samples)

            # Extract positive scores (diagonal for this chunk)
            # For chunk starting at i, positive pairs are at index i:end_i
            log_pos = t_pairs[:, :, i:end_i]
            log_pos = torch.diagonal(log_pos, dim1=1, dim2=2)  # (n_z, chunk_size_i)

            # Compute denominator
            log_denom = torch.logsumexp(t_pairs, dim=2)  # (n_z, chunk_size_i)

            # Store log ratios for this chunk
            log_ratios[:, i:end_i] = log_pos - log_denom

        # Average over samples
        mi_per_z = log_ratios.mean(dim=1)

        return mi_per_z
