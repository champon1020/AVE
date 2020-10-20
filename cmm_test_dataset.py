"""Audio visual distance learning dataset class for testing

This module provides test dataset class for audio visual distance learning.

"""
from typing import Dict

from torch import Tensor
from torch.utils.data import Dataset

from ave import AVE


class CMMTestDataset(AVE, Dataset):
    """AVE dataset class for testing distance learning

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
