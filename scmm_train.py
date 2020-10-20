"""Train module for audio visual distance learnint

This module provides training or validation functions.

"""

from torch.utils.data import DataLoader

from scmm_dataset import SCMMDataset
from scmm_model import AVDLN


class Training:
    """Training class for audio visual distance learning

    Attributes:
        model (AVDLN): audio visual distance learning model.
        train_ds (SCMMDataset): training dataset.
        valid_ds (SCMMDataset): validation dataset.
        batch_size (int): batch size.
        epoch (int): epoch size.
    """

    def __init__(
        self,
        model: AVDLN,
        train_ds: SCMMDataset,
        valid_ds: SCMMDataset,
        batch_size: int,
        epoch: int,
    ):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.epoch = epoch

    def train(self):
        """Traiinig function"""
        train_loader = DataLoader(self.train_ds, self.batch_size, shuffle=True)

        for ep in range(self.epoch):
            for batch in train_loader:
                print(batch)
                exit(1)
