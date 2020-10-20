"""Some functions for audio visual retrieval

This module provides some functions for A2V or V2A retrieval.

"""

import numpy as np
import torch
from torch import Tensor

from cmm_dataset import CMMDataset
from cmm_model import AVDLN


def euclidean_distance(vec1: Tensor, vec2: Tensor) -> Tensor:
    """Calculate euclidean distance between two vectors.

    Args:
        vec1 (torch.Tensor): tensor, [batch, dim].
        vec2 (torch.Tensor): tensor, [batch, dim].

    Returns:
        torch.Tensor: euclidean distance with tensor, [batch].

    """
    return torch.sum(torch.pow(vec1 - vec2, 2.0), dim=-1)


def a2v(model: AVDLN, cmm_ds: CMMDataset, iternum: int) -> float:
    """Retrieval of audio to video

    Args:
        model (AVDLN): network class.
        cmm_ds (CMMDataset): dataset class.
        iternum (int): the nubmer of loop iteration.

    Returns:
        float: accuracy.

    """
    frame_num = cmm_ds.frame_num
    dataset_len = len(cmm_ds) // frame_num
    iterarray = np.random.randint(0, dataset_len, iternum)

    correct_num = 0

    # Loop by dataset video.
    # This loop is replacement for the batch loop.
    for idx in iterarray:
        label = cmm_ds.annotations[idx]
        start_idx = label["start_time"] + idx * frame_num
        query_length = label["end_time"] - label["start_time"]

        # Prepare the query of audio segment.
        audio_segment = cmm_ds.get_audio_segment(start_idx, query_length)
        video_segment = None

        pred_start_idx = 0
        min_dist_sum = 1e18

        # Loop for start index.
        for video_idx in range(len(cmm_ds) - query_length):
            # If video_idx is 0, video_segment is None.
            # So it creates new video segment.
            # If video_idx is greater than 0, video segment has already existed.
            # So drop first dimension and add new video frame to the tail of the segment.
            if video_idx == 0:
                video_segment = cmm_ds.get_video_segment(video_idx, query_length)
            else:
                video_frame = cmm_ds.get_video_frame(
                    video_idx + query_length
                ).unsqueeze(0)
                video_segment = torch.cat((video_segment[1:], video_frame), dim=0)

            # Sum of euclidean distance.
            dist_sum = torch.as_tensor(0.0).cuda()

            # Loop by query length.
            for i in range(query_length):
                audio = audio_segment[i].unsqueeze(0).cuda()
                video = video_segment[i].unsqueeze(0).cuda()

                # Get audio and video embedding.
                embed_audio, embed_video = model(audio, video)

                # Calculate euclidean distance.
                dist = euclidean_distance(embed_audio, embed_video)

                dist_sum += dist.squeeze(0)

            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                pred_start_idx = video_idx

        if pred_start_idx == start_idx:
            correct_num += 1

    # Calculate validation accuracy and append it to list.
    acc = correct_num / iternum

    return acc


def v2a(model: AVDLN, cmm_ds: CMMDataset, iternum: int) -> float:
    """Retrieval of video to audio

    Args:
        model (AVDLN): network class.
        cmm_ds (CMMDataset): dataset class.
        iternum (int): the nubmer of loop iteration.

    Returns:
        float: accuracy.

    """
    frame_num = cmm_ds.frame_num
    dataset_len = len(cmm_ds) // frame_num
    iterarray = np.random.randint(0, dataset_len, iternum)

    correct_num = 0

    # Loop by dataset video.
    # This loop is replacement for the batch loop.
    for idx in iterarray:
        label = cmm_ds.annotations[idx]
        start_idx = label["start_time"] + idx * frame_num
        query_length = label["end_time"] - label["start_time"]

        # Prepare the query of audio segment.
        audio_segment = None
        video_segment = cmm_ds.get_video_segment(start_idx, query_length)

        pred_start_idx = 0
        min_dist_sum = 1e18

        # Loop for start index.
        for audio_idx in range(len(cmm_ds) - query_length):
            # If video_idx is 0, video_segment is None.
            # So it creates new video segment.
            # If video_idx is greater than 0, video segment has already existed.
            # So drop first dimension and add new video frame to the tail of the segment.
            if audio_idx == 0:
                audio_segment = cmm_ds.get_audio_segment(audio_idx, query_length)
            else:
                audio_frame = cmm_ds.get_audio_frame(
                    audio_idx + query_length
                ).unsqueeze(0)
                audio_segment = torch.cat((audio_segment[1:], audio_frame), dim=0)

            # Sum of euclidean distance.
            dist_sum = torch.as_tensor(0.0).cuda()

            # Loop by query length.
            for i in range(query_length):
                audio = audio_segment[i].unsqueeze(0).cuda()
                video = video_segment[i].unsqueeze(0).cuda()

                # Get audio and video embedding.
                embed_audio, embed_video = model(audio, video)

                # Calculate euclidean distance.
                dist = euclidean_distance(embed_audio, embed_video)

                dist_sum += dist.squeeze(0)

            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                pred_start_idx = audio_idx

        if pred_start_idx == start_idx:
            correct_num += 1

    # Calculate validation accuracy and append it to list.
    acc = correct_num / iternum

    return acc
