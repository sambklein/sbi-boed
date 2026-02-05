# Adapted from ALINE codebase
# https://github.com/huangdaolang/ALINE/blob/35768b17d1c836e8686dbbea48e863ebb962a7b8/loss/eig.py#L153
import torch
import torch.nn as nn


class Pce(nn.Module):
    def __init__(self, L: int, M: int, log_prob, reduction=None) -> None:
        """Base class for EIG Bounds, compute at each step

        Args:
            L (int): number of contrastive samples $\theta_{1:L}$
            M (int): number of parallel trajectories, i.e. batch_size
            log_prob (function): function to calculate the log likelihoods given theta and design policy
        """
        super(Pce, self).__init__()
        self.L = L
        self.M = M
        self.log_prob = log_prob
        self.reduction = reduction
        self.seq_logprobs = torch.zeros((L + 1, M))

    def reset(self):
        """Reset the sequential log likelihood"""
        self.seq_logprobs = torch.zeros((self.L + 1, self.M))

    def step(self, y_outcomes, xi_designs, thetas):
        """Accumulate the sequential log likelihood

        Args:
            y_outcomes [M, D_y]: a batch of sequential experiment's outcomes
            xi_designs [M, D_x]: a batch of designs
            thetas [L, M, (K, )D]: L batches of latent variables sampled from the prior

        Returns:
            log_probs [L, M]
        """

        xi_designs = xi_designs.unsqueeze(0)  # [1, B, D_x]
        y_outcomes = y_outcomes.unsqueeze(0)  # [1, B, D_y]
        batch_size = y_outcomes.shape[1]

        log_probs = self.log_prob(y_outcomes, xi_designs, thetas).squeeze(-1)  # [L, B]

        # accumulate the sequential joint log likelihood
        self.seq_logprobs += log_probs.view(-1, batch_size)  # [L, B]

        return self.seq_logprobs

    def forward(self, y_outcomes, xi_designs, thetas):
        log_probs = self.step(y_outcomes, xi_designs, thetas)  # [L+1, M]

        # Compute log-ratios relative to true theta (index 0) for numerical stability
        # log(p(y_{1:t}|theta_l) / p(y_{1:t}|theta_0))
        log_ratios = log_probs - log_probs[:1]  # [L+1, M]

        # Lower bound: logsumexp of ratios (note: log_ratios[0] = 0, so sum >= 1)
        # sPCE = log(L+1) - logsumexp(log_ratios)
        pce_loss = torch.logsumexp(log_ratios, dim=0)  # [M]

        # Upper bound (NMC): exclude true theta from sum
        nmc_loss = torch.logsumexp(log_ratios[1:], dim=0)  # [M]

        # Outer expectation
        if self.reduction == "mean":
            pce_loss = torch.mean(pce_loss)
            nmc_loss = torch.mean(nmc_loss)

        return pce_loss, nmc_loss


@torch.no_grad()
def compute_eig(prior, log_likelihood, theta_0, x, y, L=int(1e7)):
    """Evaluate the lower and upper bounds of EIG from a minibatch of the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]
    batch_size = x.shape[0]

    criterion = Pce(L, batch_size, log_likelihood, reduction="none")

    pce_losses = []
    # Sometimes the prior has a different sampling method for evaluation
    # see for example source_finding (eval_sample matches existing implementations)
    sample_method = getattr(prior, "eval_sample", prior.sample)
    thetas = sample_method((L * batch_size,)).view(L, batch_size, -1).cpu()
    thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)  # [L+1, B, (K, )D]

    for t in range(T):
        # Accumulate the EIG losses at each step (internal state maintained in criterion)
        pce_loss, _ = criterion(y[:, t], x[:, t], thetas)  # [B]
        pce_losses.append(pce_loss)

    pce_losses = torch.stack(pce_losses, dim=-1)  # [B, T]

    # Calculate bounds
    pce_losses = torch.log(torch.tensor(L + 1)) - pce_losses  # [B(, T)]

    return pce_losses
