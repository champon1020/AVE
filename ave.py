"""AVE dataset abstract class

This module provides abstract class for AVE dataset.

"""

from abc import ABCMeta
from typing import List

import numpy as np
import torch


class AVE(metaclass=ABCMeta):
    """AVE dataset loader class

    Attributes:
        annot_path (str): annotation file path.
        frame_num (int): the number of frames.
        target_size (int): the number of categories included in AVE dataset.
        annotations (Dict[]): all annotations list.

    """

    def __init__(
        self,
        annot_path: str,
        target_size: int,
    ):
        self.frame_num = 10
        self.annot_path = annot_path
        self.target_size = target_size

        self.annotations = []
        self.category_dict = {"None": self.target_size - 1}
        self._load_annot()

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
        segment = [
            self.category_dict[category]
            if start_time <= t or t < end_time
            else self.category_dict["None"]
            for t in range(self.frame_num)
        ]

        return torch.from_numpy(np.array(segment))
