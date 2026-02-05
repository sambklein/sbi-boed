"""Function to learn the mean of a stochastic."""

import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from torch import nn
import torch.nn.functional as F

from sbi_bax.models.mlp import Mlp

log = logging.getLogger(__name__)


class PotentialNetwork(nn.Module):
    """Neural network to learn potential function over designs."""

    def __init__(self, design_dim, hidden_dim=12, buffer_size=1_00, lr=1e-3):
        super().__init__()
        self.network = Mlp(design_dim, 1, hidden_dims=[hidden_dim, hidden_dim])
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)

        # Buffer to store (design, loss) pairs
        self.design_buffer = []
        self.loss_buffer = []
        self.buffer_size = buffer_size

    def add_samples(self, designs, losses):
        """Add samples to buffer for training."""
        self.design_buffer.append(designs.detach().cpu())
        self.loss_buffer.append(losses.detach().cpu())

        # Keep only recent samples
        if len(self.design_buffer) > self.buffer_size:
            self.design_buffer.pop(0)
            self.loss_buffer.pop(0)

    def train_on_buffer(self, n_epochs=10, batch_size=256):
        """Train potential network on buffered samples."""
        if len(self.design_buffer) == 0:
            return 0.0

        # Concatenate buffer
        all_designs = torch.cat(self.design_buffer, dim=0)
        all_losses = torch.cat(self.loss_buffer, dim=0)

        dataset = TensorDataset(all_designs, all_losses)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        for _ in range(n_epochs):
            for designs_batch, losses_batch in loader:
                designs_batch = designs_batch.to(next(self.network.parameters()).device)
                losses_batch = losses_batch.to(next(self.network.parameters()).device)

                pred = self.network(designs_batch).squeeze()
                loss = F.mse_loss(pred, losses_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (n_epochs * len(loader))

    def forward(self, designs):
        return self.network(designs).squeeze()
