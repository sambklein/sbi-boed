# Simple transformer encoder for encoding datasets of samples.
import torch
import torch.nn as nn

from sbi_bax.models.linear import LinearEncoder


class TransformerEmbeddor(nn.Module):
    """Encodes individual trials and applies transformer layers."""

    def __init__(
        self,
        input_dim: int,
        trial_net_output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_sequence_length: int = 1000,
    ):
        super().__init__()
        self.encoder = LinearEncoder(
            input_dim=input_dim, output_dim=trial_net_output_dim
        )
        self.trial_net_output_dim = trial_net_output_dim
        self.use_positional_encoding = use_positional_encoding

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(max_sequence_length + 2, trial_net_output_dim) * 0.1
            )

        # Multiple transformer layers with proper FFN
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=trial_net_output_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

    def fit_norm(self, data):
        """Fit normalization of the encoder if it has a norm method."""
        if hasattr(self.encoder, "fit_norm"):
            batch_size, num_trials, _ = data.shape
            self.encoder.fit_norm(data.view(batch_size * num_trials, -1))

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_trials, trial_dim]
        Returns:
            hidden: [batch_size, num_trials, trial_net_output_dim]
        """
        batch_size, num_trials, _ = x.shape

        # Encode each trial
        x_flat = x.view(batch_size * num_trials, -1)
        trial_embeddings = self.encoder(x_flat.unsqueeze(1)).squeeze(1)
        trial_embeddings = trial_embeddings.view(
            batch_size, num_trials, self.trial_net_output_dim
        )

        # Add positional encoding
        if self.use_positional_encoding:
            seq_len = trial_embeddings.size(1)
            pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)
            trial_embeddings = trial_embeddings + pos_enc

        # Apply transformer layers
        hidden = trial_embeddings
        for layer in self.transformer_layers:
            hidden = layer(hidden)

        return hidden


class TransformerPoolingHead(nn.Module):
    """Pools transformer embeddings and projects to output dimension."""

    def __init__(
        self,
        trial_net_output_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learned aggregation instead of simple mean
        self.attention_pooling = nn.MultiheadAttention(
            trial_net_output_dim, num_heads=num_heads, batch_first=True
        )
        self.pooling_query = nn.Parameter(torch.randn(1, trial_net_output_dim))

        # Output projection with more capacity
        self.output_projection = nn.Sequential(
            nn.Linear(trial_net_output_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
        )

    def forward(self, hidden):
        """
        Args:
            hidden: [batch_size, num_trials, trial_net_output_dim]
        Returns:
            output: [batch_size, output_dim]
        """
        batch_size = hidden.size(0)

        # Learned attention pooling
        query = self.pooling_query.expand(batch_size, -1, -1)  # (batch, 1, dim)
        pooled, _ = self.attention_pooling(query, hidden, hidden)
        pooled = pooled.squeeze(1)  # (batch, dim)

        # Output projection
        output = self.output_projection(pooled)
        return output


class TransformerEncoder(nn.Module):
    """Full transformer encoder: embeddor + pooling head."""

    def __init__(
        self,
        input_dim: nn.Module,
        trial_net_output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 10,
        use_positional_encoding: bool = False,
        max_sequence_length: int = 1000,
        pooling_num_heads: int = 4,
    ):
        super().__init__()

        self.embeddor = TransformerEmbeddor(
            input_dim=input_dim,
            trial_net_output_dim=trial_net_output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            max_sequence_length=max_sequence_length,
        )

        self.pooling_head = TransformerPoolingHead(
            trial_net_output_dim=trial_net_output_dim,
            output_dim=output_dim,
            num_heads=pooling_num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def fit_norm(self, data):
        """Fit normalization of the encoder if it has a norm method."""
        self.embeddor.fit_norm(data)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_trials, trial_dim]
        Returns:
            output: [batch_size, output_dim]
        """
        hidden = self.embeddor(x)
        output = self.pooling_head(hidden)
        return output
