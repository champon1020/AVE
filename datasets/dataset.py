"""Audio visual event localization dataset

This module provides dataset class for audio visual event localization.

"""

import os
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .ave import AVE


class AVELDataset(AVE, Dataset):
    """AVE dataset class for event localization

    Attributes:
        annotation_path: (str): Annotation file path.
        target_size (int): Number of categories included in AVE dataset.
        features_path (str): Audio and visual features directory path.

    """

    def __init__(
        self,
        annotation_path: str,
        target_size: int,
        features_path: str,
        n_frames: int,
    ):
        super().__init__(annotation_path, target_size, n_frames)
        self.features_path = features_path

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id = self.annotations[idx]["video_id"]
        feature_name = "{0}.pt".format(video_id)
        feature_a = torch.load(os.path.join(self.features_path, "audio", feature_name))
        feature_v = torch.load(os.path.join(self.features_path, "visual", feature_name))

        sample = {
            "audio": feature_a,
            "visual": feature_v,
            "label": self.annotations[idx],
        }

        return sample
