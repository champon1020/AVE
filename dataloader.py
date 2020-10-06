"""Dataloader module

This module provides AVE dataset loader class.

"""
import os

import torch
import torch.utils.data as data


class AVEDataset(data.Dataset):
    """AVE dataset loader class

    Attributes:
        ave_root (str): ave dataset root directory path.
        annot_path (str): annotation file path.
        batch_size (int): dataset batch size.

    """

    def __init__(self, ave_root, annot_path, batch_size):
        self.ave_root = ave_root
        self.annot_path = annot_path
        self.batch_size = batch_size
        self.annotations = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embed_name = "{0}.pt".format(idx)
        feature_a = torch.load(os.path.join("features/audio", embed_name))
        feature_v = torch.load(os.path.join("features/frame", embed_name))

        sample = {"audio": feature_a, "video": feature_v}
        return sample

    def _load_annot(self):
        """
        Load AVE Dataset Annotation

        """
        with open(self.annot_path) as f:
            for line in f:
                annots = line.split("&")
                self.annotations.append(
                    Annotation(annots[0], annots[1], annots[3], annots[4])
                )


class Annotation:
    """AVE annotation class

    Attributes:
        category (str): category annotation.
        video_id (str): unique id.
        start_time (int): event start time.
        end_time (int): event end time.

    """

    def __init__(self, category, video_id, start_time, end_time):
        self.cateogry = category
        self.video_id = video_id
        self.start_time = start_time
        self.end_time = end_time

    def equal(self, video_id):
        """Is Equal

        Return the bool variable whether this annotation has the video_id of first argument.

        Args:
            video_id (int): unique id for all data.

        Returns:
            bool: whether this annotation has the video_id or not.

        """
        return self.video_id == video_id
