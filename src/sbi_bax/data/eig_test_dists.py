#!/usr/bin/env python3
"""
Test script for RqsPureContextFlow on a simple multimodal distribution.
"""

import torch
import numpy as np


from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal


class MultimodalDistribution:
    def __init__(self, means, covariances, weights):
        """
        Create mixture of Gaussians with known entropy.

        Args:
            means: List of mean vectors [(d,), (d,), ...]
            covariances: List of covariance matrices [(d,d), (d,d), ...]
            weights: Mixture weights [w1, w2, ...]
        """
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.covariances = [torch.tensor(c, dtype=torch.float32) for c in covariances]
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize

        # Create the mixture distribution
        mix = Categorical(self.weights)
        comp = MultivariateNormal(
            torch.stack(self.means), torch.stack(self.covariances)
        )
        self.distribution = MixtureSameFamily(mix, comp)

        # Compute exact entropy
        self.estimated_entropy = self._estimate_entropy()

    def _estimate_entropy(self):
        """
        Estimate entropy with MC.
        """
        n_samples = 1_000_000
        # Sample from the distribution
        samples = self.distribution.sample([n_samples])
        # Compute the log probability
        unnorm_log_prob = self.distribution.log_prob(samples)
        # # Get the partition
        # partition = torch.logsumexp(unnorm_log_prob, -1)
        # # Compute the entropy
        # return - torch.mean(unnorm_log_prob - partition)
        return -torch.mean(unnorm_log_prob)

    def sample(self, n_samples):
        """Sample from the mixture."""
        return self.distribution.sample((n_samples,))

    def log_prob(self, x):
        """Log probability under the mixture."""
        return self.distribution.log_prob(x)


class GaussianToMultiModal:
    def __init__(
        self,
        mean_current,
        std_current,
        mean_updated,
        covariance_updated,
        weights,
        n_samples,
    ):
        self.mean_current = torch.tensor(mean_current, dtype=torch.float32)
        self.std_current = torch.tensor(std_current, dtype=torch.float32)
        self.mean_updated = torch.tensor(mean_updated, dtype=torch.float32)
        self.covariance_updated = torch.tensor(covariance_updated, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.n_samples = n_samples
        self.n_dim = self.mean_current.shape[0]
        self.updated_dist = MultimodalDistribution(
            mean_updated, covariance_updated, weights
        )
        assert (
            self.n_dim
            == self.std_current.shape[0]
            == self.mean_updated.shape[0]
            == self.covariance_updated.shape[0]
        )
        # Generate sample from current posterior
        self.current_sample = self.sample(
            self.mean_current, self.std_current, n_samples
        )
        self.updated_sample = self.updated_dist.sample(n_samples)
        # Compute the exact entropy of each of these samples
        self.current_entropy = (
            0.5 * (1 + torch.log(2 * torch.pi * self.std_current**2)).sum()
        )
        self.updated_entropy = self.updated_dist.estimated_entropy
        # Get the EIG from this update
        self.eig = self.current_entropy - self.updated_entropy

    def sample(self, mean, variance, n_samples=1):
        """Generate samples from the current posterior."""
        return torch.Tensor(
            np.random.normal(
                loc=mean.cpu().numpy(),
                scale=variance.cpu().numpy(),
                size=(n_samples, self.n_dim),
            )
        )


class GaussianToGaussian:
    def __init__(self, mean_current, std_current, mean_updated, std_updated, n_samples):
        self.mean_current = mean_current
        self.std_current = std_current
        self.mean_updated = mean_updated
        self.std_updated = std_updated
        self.n_samples = n_samples
        self.n_dim = mean_current.shape[0]
        assert (
            self.n_dim
            == std_current.shape[0]
            == mean_updated.shape[0]
            == std_updated.shape[0]
        )
        # Generate sample from current posterior
        self.current_sample = self.sample(
            self.mean_current, self.std_current, n_samples
        )
        self.updated_sample = self.sample(
            self.mean_updated, self.std_updated, n_samples
        )
        # Compute the exact entropy of each of these samples
        self.current_entropy = (
            0.5 * (1 + torch.log(2 * torch.pi * std_current**2)).sum()
        )
        self.updated_entropy = (
            0.5 * (1 + torch.log(2 * torch.pi * std_updated**2)).sum()
        )
        # Get the EIG from this update
        self.eig = self.current_entropy - self.updated_entropy

    def sample(self, mean, variance, n_samples=1):
        """Generate samples from the current posterior."""
        return torch.Tensor(
            np.random.normal(
                loc=mean.cpu().numpy(),
                scale=variance.cpu().numpy(),
                size=(n_samples, self.n_dim),
            )
        )


class MineDataset:
    # This is the test distribution used in MINE
    def __init__(self, n_samples, dim, rho):
        self.n_samples = n_samples
        self.rho = rho
        self.dim = dim

        self.dist = self.build_dist
        self.x = self.dist.sample((n_samples,))
        self.dim = dim
        # Split off the current and updated samples
        self.current_sample = self.x[:, : self.dim]
        self.updated_sample = self.x[:, self.dim :]

    @property
    def build_dist(self):
        mu = torch.zeros(2 * self.dim)
        dist = MultivariateNormal(mu, self.cov_matrix)
        return dist

    @property
    def cov_matrix(self):
        cov = torch.zeros((2 * self.dim, 2 * self.dim))
        cov[torch.arange(self.dim), torch.arange(start=self.dim, end=2 * self.dim)] = (
            self.rho
        )
        cov[torch.arange(start=self.dim, end=2 * self.dim), torch.arange(self.dim)] = (
            self.rho
        )
        cov[torch.arange(2 * self.dim), torch.arange(2 * self.dim)] = 1.0
        return cov

    @property
    def eig(self):
        # return -0.5 * np.log(np.linalg.det(self.cov_matrix.data.numpy()))
        return -np.log(1 - self.rho**2) * self.dim / 2
