"""Dataloader module

This module provides AVE dataset loader class.

"""
import os
from typing import Dict, List

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
        frame_num (int): the number of frames.
        target_size (int): the number of categories included in AVE dataset.

    """

    def __init__(
        self, ave_root: str, annot_path: str, features_path: str, batch_size: int
    ):
        self.ave_root = ave_root
        self.annot_path = annot_path
        self.features_path = features_path
        self.batch_size = batch_size
        self.frame_num = 10
        self.target_size = 29

        self.annotations = []
        self.category_dict = {"None": 28}
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
            category_num = 0
            iterlines = iter(f)
            next(iterlines)

            for line in iterlines:
                # Annotations format is bellow.
                # <category>&<video_id>&<video_quality>&<event_start_time>&<event_end_time>
                annots = line.split("&")

                # If category name is not exist in category_dict, add category as key
                # and iterate the category number.
                if self.category_dict.get(annots[0]) is None:
                    self.category_dict[annots[0]] = category_num
                    category_num += 1

                label = self._generate_segment_label(
                    annots[0], int(annots[3]), int(annots[4].split("\n")[0])
                )

                self.annotations.append(
                    {"category": annots[0], "video_id": annots[1], "label": label}
                )

    def _generate_segment_label(
        self, category: str, start_time: int, end_time: int
    ) -> List[int]:
        """Generate time segment label

        Args:
            start_time (List[int]): event start times.
            end_time (List[int]): event end times.

        """
        segment = torch.zeros(self.frame_num, self.target_size)

        # Initialize all segment as category "None".
        segment[:, -1] = 1

        for i in range(start_time, end_time):
            category_idx = self.category_dict[category]
            segment[i, category_idx] = 1
            segment[i, -1] = 0

        return segment
