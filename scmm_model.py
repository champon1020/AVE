"""Audio visual distance learning network

This module provides network class for distance learning.

"""

import torch.nn as nn


class AVDLN(nn.Module):
    """Audio visual distance learning network.

    Attributes:
        audio_input (int): audio input size of fc layer.
        video_input (int): video input size of fc layer.
        hidden_size (int): hidden size of fc layer.
        output_size (int): output size of fc layer.

    """

    def __init__(
        self, audio_input: int, video_input: int, hidden_size: int, output_size: int
    ):
        super().__init__()
        self.dense_audio = nn.Sequential(
            nn.Linear(audio_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.dense_video = nn.Sequential(
            nn.Linear(video_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, audio, video):
        """Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, audio_dim].
            video (torch.Tensor): video feature, [batch, video_dim, ?].

        """
        audio = self.dense_audio(audio)
        video = self.dense_video(video)
