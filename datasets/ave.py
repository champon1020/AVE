"""
This module provides abstract class for AVE dataset.

"""

from abc import ABCMeta
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class AVE(metaclass=ABCMeta):
    """AVE dataset loader class.

    Attributes:
        annot_path (str): Annotation file path.
        target_size (int): Number of categories included in AVE dataset.
        n_frames (int): Number of frames.
        annotations (Dict[]): All annotations list.

    """

    def __init__(self, annotation_path: str, target_size: int, n_frames: int):
        self.annot_path = annotation_path
        self.target_size = target_size
        self.n_frames = n_frames

        self.annotations = []
        self.category_dict = {"None": self.target_size - 1}
        self._load_annot()

    def _load_annot(self):
        """
        Load AVE Dataset Annotation.

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

                category = annots[0]
                start_time = int(annots[3])
                end_time = int(annots[4].split("\n")[0])

                # label = self._generate_segment_label(category, start_time, end_time)
                onehot_label = self._generate_onehot_segment_label(
                    category,
                    start_time,
                    end_time,
                )

                self.annotations.append(
                    {
                        "video_id": annots[1],
                        "onehot_label": onehot_label,
                        "category": category,
                    }
                )

    def _generate_segment_label(
        self, category: str, start_time: int, end_time: int
    ) -> List[int]:
        """Generate time segment label.

        Example:
            If category is "ChurchBell" which and timestamp is "3" to "9",
            return the tensor, [28, 28, 0, 0, 0, 0, 0, 0, 0, 28].
            The number means the category index.

        Args:
            start_time (List[int]): Event start times.
            end_time (List[int]): Event end times.

        Returns:
            torch.Tensor: Segment label tensor.

        """
        segment = [
            self.category_dict[category]
            if start_time <= t < end_time
            else self.category_dict["None"]
            for t in range(self.n_frames)
        ]

        return torch.from_numpy(np.array(segment))

    def _generate_onehot_segment_label(
        self,
        category: str,
        start_time: int,
        end_time: int,
    ) -> List[int]:
        category = torch.as_tensor(self.category_dict[category])
        category_none = torch.as_tensor(self.category_dict["None"])

        segment = [
            F.one_hot(category, num_classes=self.target_size).numpy()
            if start_time <= t < end_time
            else F.one_hot(category_none, num_classes=self.target_size).numpy()
            for t in range(self.n_frames)
        ]

        return torch.from_numpy(np.array(segment))
