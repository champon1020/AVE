"""Dataloader module

This module provides AVE dataset loader class.

"""
import os
from typing import Dict

import torch
import torch.utils.data as data
from torch import Tensor


class AVEDataset(data.Dataset):
    """AVE dataset loader class

    Attributes:
        ave_root (str): ave dataset root directory path.
        annot_path (str): annotation file path.
        batch_size (int): dataset batch size.
        annotations (Dict[]): all annotations list.

    """

    def __init__(
        self, ave_root: str, annot_path: str, features_path: str, batch_size: int
    ):
        self.ave_root = ave_root
        self.annot_path = annot_path
        self.features_path = features_path
        self.batch_size = batch_size

        self.annotations = []
        self._load_annot()

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

    def _load_annot(self):
        """
        Load AVE Dataset Annotation

        """
        with open(self.annot_path) as f:
            for line in f:
                annots = line.split("&")
                self.annotations.append(
                    {
                        "category": annots[0],
                        "video_id": annots[1],
                        "start_time": annots[2],
                        "end_time": annots[3],
                    }
                )
