"""Audio visual distance learning dataset

This module provides dataset for audio visual distance learning.

"""
import os
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ave import AVE


class CMMDataset(AVE, Dataset):
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
        frame_idx = idx % self.frame_num

        rand_idx = self._get_random_idx(idx)
        rand_video_idx = rand_idx // self.frame_num
        rand_frame_idx = rand_idx % self.frame_num

        if torch.is_tensor(video_idx):
            video_idx = video_idx.tolist()

        if torch.is_tensor(frame_idx):
            frame_idx = frame_idx.tolist()

        if torch.is_tensor(rand_video_idx):
            rand_video_idx = rand_video_idx.tolist()

        if torch.is_tensor(rand_frame_idx):
            rand_frame_idx = rand_frame_idx.tolist()

        audio_video_id = self.annotations[video_idx]["video_id"]
        frame_video_id = self.annotations[rand_video_idx]["video_id"]

        audio_feature_name = "{0}.pt".format(audio_video_id)
        frame_feature_name = "{0}.pt".format(frame_video_id)

        feature_a = torch.load(
            os.path.join(self.features_path, "audio", audio_feature_name)
        )
        feature_v = torch.load(
            os.path.join(self.features_path, "frame", frame_feature_name)
        )

        sample = {
            "audio": feature_a[frame_idx, :],
            "video": feature_v[frame_idx, :, :, :],
            "label": 1
            if video_idx == rand_video_idx and frame_idx == rand_frame_idx
            else 0,
        }

        return sample

    def _get_random_idx(self, idx: int) -> int:
        """Get random index.

        This methods returns index number.
        If the rand is greater than 0.5, this returns the same number to argument.
        If the rand is less than 0.5, this returns the random number the range
        between 0 to total length.

        Args:
            idx (int): index nubmer.

        Returns:
            int: random index number.

        """
        if np.random.rand() < 0.5:
            rand = np.random.randint(0, self.__len__())
            return (idx + rand) % self.__len__()

        return idx
