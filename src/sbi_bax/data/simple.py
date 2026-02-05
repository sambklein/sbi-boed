from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, y, ctx):
        """
        Args:
            y (torch.Tensor): The target values (e.g., outputs).
            ctx (torch.Tensor): The context values (e.g., inputs or additional data).
        """
        self.y = y
        self.ctx = ctx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], self.ctx[idx]
