"""Audio visual distance learning dataset

This module provides dataset for audio visual distance learning.

"""
import os
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ave import AVE


class SCMMDataset(AVE, Dataset):
    """AVE dataset class for distance learning

    Attributes:
        ave_root (str): ave dataset root directory path.
        annot_path (str): annotation file path.
        batch_size (int): dataset batch size.
        annotations (Dict[]): all annotations list.
        frame_num (int): the number of frames.
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
        return len(self.annotations) * self.frame_num

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        video_idx = idx // self.frame_num
        frame_idx = idx & self.frame_num

        if torch.is_tensor(video_idx):
            video_idx = video_idx.tolist()

        if torch.is_tensor(frame_idx):
            frame_idx = frame_idx.tolist()

        video_id = self.annotations[video_idx]["video_id"]
        embed_name = "{0}.pt".format(video_id)
        feature_a = torch.load(os.path.join(self.features_path, "audio", embed_name))
        feature_v = torch.load(os.path.join(self.features_path, "frame", embed_name))

        sample = {
            "audio": feature_a[frame_idx, :],
            "video": feature_v[frame_idx, :, :, :],
            "label": self.annotations[idx],
        }

        return sample
