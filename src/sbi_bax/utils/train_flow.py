# SB+ BAX for crystal diffraction images
from itertools import islice
import logging
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import torch
from nflows.flows import Flow

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# Set up logging
log = logging.getLogger(__name__)


def build_scheduler(optimizer, lr, max_epochs, frac_warmup, batches_per_epoch):
    """Build a per-batch scheduler."""
    total_steps = max_epochs * batches_per_epoch
    warmup_steps = int(frac_warmup * total_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps
        if warmup_steps > 0
        else 1.0,  # Start from lr/warmup_steps
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.1
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def train_flow(
    flow: Flow,
    data: torch.Tensor,
    condition: torch.Tensor | None = None,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_epochs: int = 100,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.2,
    work_dir: Path = Path("measurements"),
    from_scratch: bool = True,
    frac_warmup: float = 0.1,
    frac_rampdown: float = 0.2,
    initial_model: Path | None = None,
    max_batches: int | None = None,
):
    # Make the directory if it doesn't already exist
    work_dir.mkdir(parents=True, exist_ok=True)
    final_model = work_dir / "model.pt"
    # If an initial_model is passed then load it
    if initial_model is not None:
        flow.load_state_dict(torch.load(initial_model))
    # Get if the model is conditional or not
    is_conditional = condition is not None
    if not is_conditional:
        # Create a dummy condition tensor
        condition = torch.zeros((data.shape[0], 1))
    if from_scratch or not final_model.exists():
        # Prepare dataset and dataloader
        dataset = TensorDataset(data, condition)
        n_val = int(len(dataset) * validation_fraction)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)

        # Calculate batches per epoch
        batches_per_epoch = len(train_loader)
        if max_batches is not None:
            batches_per_epoch = min(batches_per_epoch, max_batches)
        # Learning rate schedule: warmup, then cosine annealing
        scheduler = build_scheduler(
            optimizer, lr, max_epochs, frac_warmup, batches_per_epoch
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        learning_rates = []

        for epoch in range(max_epochs):
            flow.train()
            train_loss = 0.0
            # Handle max_batches if specified
            batch_iter = (
                islice(train_loader, max_batches) if max_batches else train_loader
            )
            train_samples = 0  # Track actual samples seen
            # Training loop
            for batch_data, batch_condition in batch_iter:
                optimizer.zero_grad()
                # Negative log likelihood loss
                log_prob = flow.log_prob(
                    batch_data, batch_condition if is_conditional else None
                )
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * batch_data.size(0)
                train_samples += batch_data.size(0)
            train_loss /= train_samples
            train_losses.append(train_loss)

            # Validation
            flow.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_condition in val_loader:
                    log_prob = flow.log_prob(
                        batch_data, batch_condition if is_conditional else None
                    )
                    loss = -log_prob.mean()
                    val_loss += loss.item() * batch_data.size(0)
            val_loss /= n_val
            val_losses.append(val_loss)
            learning_rates.append(optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                torch.save(flow.state_dict(), work_dir / "best_flow.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= stop_after_epochs:
                    print("Early stopping triggered.")
                    break

        # Plot training curve with learning rate as a subfigure
        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax[0].plot(epochs, train_losses, label="Train Loss")
        ax[0].plot(epochs, val_losses, label="Val Loss")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Training Curve")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(epochs, learning_rates, label="LR", color="C2")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")
        ax[1].set_yscale("log")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(work_dir / "flow_training.png", bbox_inches="tight")
        plt.close(fig)

        # Copy best model to final model location
        shutil.copy2(work_dir / "best_flow.pt", final_model)
    flow.load_state_dict(torch.load(final_model))
    return flow
