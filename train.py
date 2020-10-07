"""Train module

This module provides training or validation functions.

"""
import sys

import torch.utils.data as data

from dataloader import AVEDataset
from model import DMRFE


def train(ds: AVEDataset, model: DMRFE, batch_size: int, epoch: int):
    """Training function

    Args:
        ds (dataloader.AVEDataset): dataset class.
        model (model.DMRFE): model class.

    """
    loader = data.DataLoader(ds, batch_size, shuffle=True)
    for _ in range(epoch):
        for batch in loader:
            print(batch)
            sys.exit(1)
