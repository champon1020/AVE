"""Audio visual event localization dataset

This module provides dataset class for audio visual event localization.

"""

import os
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ave import AVE


class AVELDataset(AVE, Dataset):
    """AVE dataset class for event localization

    Attributes:
        ave_root (str): ave dataset root directory path.
        batch_size (int): dataset batch size.
        target_size (int): the number of categories included in AVE dataset.

    """

    def __init__(
        self,
        ave_root: str,
        annot_path: str,
        features_path: str,
        batch_size: int,
        target_size: int,
    ):
        super().__init__(annot_path, target_size)
        self.ave_root = ave_root
        self.features_path = features_path
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id = self.annotations[idx]["video_id"]
        embed_name = "{0}.pt".format(video_id)
        feature_a = torch.load(os.path.join(self.features_path, "audio", embed_name))
        feature_v = torch.load(os.path.join(self.features_path, "frame", embed_name))

        sample = {
            "audio": feature_a,
            "video": feature_v,
            "label": self.annotations[idx],
        }

        return sample
