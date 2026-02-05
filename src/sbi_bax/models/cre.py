import torch

from sbi_bax.models.mlp import Mlp


class MlpClassifier(torch.nn.Module):
    def __init__(
        self,
        theta_dim: int,
        obs_dim: int,
        mlp_kwargs: dict,
    ):
        super().__init__()
        self.network = Mlp(
            input_dim=theta_dim + obs_dim,
            output_dim=1,
            **mlp_kwargs,
        )

    def fit_norm(self, theta_data, obs_data):
        # Concatenate theta and x for normalization
        tx_data = torch.cat([theta_data, obs_data], dim=-1)
        self.network.fit_norm(tx_data)

    def forward(self, theta, x):
        # Concatenate theta and x
        tx = torch.cat([theta, x], dim=-1)
        logits = self.network(tx).squeeze(-1)
        return logits
