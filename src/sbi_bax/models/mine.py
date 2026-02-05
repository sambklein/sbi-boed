import torch
import torch.nn as nn
from sbi_bax.models.mlp import Mlp


# ----------------------------
# Conditional MINE class
# ----------------------------
class ConditionalMine(nn.Module):
    """
    Conditional MINE estimator for I(X;Y | Z=z).
    Assumes input batches are sampled from p(x,y|z) for a fixed z
    (or that user handles grouping by z outside).
    The critic T_theta(x,y,z) is an MLP that receives concatenated features.
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        z_dim,
        hidden=(256, 256),
        ema_decay=0.99,
        moving_average=False,
    ):
        """
        Args:
            x_dim, y_dim, z_dim: input dimensions
            hidden: tuple of hidden layer sizes for critic
            ema_decay: decay rate for exponential moving average of exp(T_neg)
            moving_average: whether to use EMA stabilization
        """
        super().__init__()
        self.in_dim = x_dim + y_dim + z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.critic = Mlp(self.in_dim, 1, hidden_dims=hidden)
        self.ema_decay = ema_decay
        self.register_buffer(
            "ma_et", torch.tensor(1.0)
        )  # moving average of E[e^{T_neg}]
        self.moving_average = moving_average

    def forward_T(self, x, y, z):
        n_z, n_samples, _ = x.shape

        out = self.critic(
            torch.cat(
                [
                    x.reshape(-1, self.x_dim),
                    y.reshape(-1, self.y_dim),
                    z.reshape(-1, self.z_dim),
                ],
                dim=1,
            )
        )

        return out.reshape(n_z, n_samples)

    @staticmethod
    def _shuffle_y(y):
        n_z, n_samples, _ = y.shape
        idx = torch.stack(
            [torch.randperm(n_samples, device=y.device) for _ in range(n_z)]
        )  # (n_z, n_samples)
        return y[torch.arange(n_z)[:, None], idx, :]

    def mi_estimate(self, x, y, z):
        """
        x, y, z: (n_z, n_samples, dim)
        Returns:
            mi: (n_z,) tensor of MI values, one per z
        """

        # t_joint: (n_z, n_samples)
        t_joint = self.forward_T(x, y, z)

        # shuffle y across sample dimension (axis=1)
        y_shuffled = self._shuffle_y(y)

        # t_marg: (n_z, n_samples)
        t_marg = self.forward_T(x, y_shuffled, z)

        # joint expectation: E[T] per z  → shape (n_z,)
        mean_t = t_joint.mean(dim=1)

        # log-mean-exp for marginals, per z
        # log(1/N sum exp(t_marg)) = logsumexp(...) - log N
        n_samples = x.shape[1]
        lme = torch.logsumexp(t_marg, dim=1) - torch.log(
            torch.tensor(n_samples, device=x.device, dtype=t_joint.dtype)
        )

        if self.moving_average:
            # EMA of exp(T) must be tracked *per z*, so store a buffer of shape (n_z,)
            et_mean = torch.exp(t_marg).mean(dim=1).detach()

            # update EMA
            self.ma_et = self.ema_decay * self.ma_et + (1 - self.ema_decay) * et_mean

            # MI per z
            mi = mean_t - torch.log(self.ma_et + 1e-12)

        else:
            mi = mean_t - lme

        return mi  # shape (n_z,)
