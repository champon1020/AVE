"""Audio visual distance learning network

This module provides network class for distance learning.

"""

import torch
import torch.nn as nn

from avel_model import AttentionNet


class AVDLN(nn.Module):
    """Audio visual distance learning network.

    Attributes:
        audio_dim (int): audio feature dimension.
        video_dim (int): video feature dimension.
        att_embed_dim (int): embedding dimension for attention network.
        hidden_dim (int): hidden size of fc layer.
        output_dim (int): output size of fc layer.

        att_network: audio-guided visual attention network.
        dense_audio: audio embedding network.
        dense_video: visual embedding netowrk.

    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        att_embed_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.att_network = AttentionNet(
            audio_dim,
            video_dim,
            7 * 7,
            att_embed_dim,
        )

        self.dense_audio = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dense_video = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, audio, video):
        """Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, audio_dim].
            video (torch.Tensor): video feature, [batch, video_dim, video_height, video_width].

        Returns:
            (torch.Tensor): audio embedding tensor, [batch, output_dim].
            (torch.Tensor): video embedding tensor, [batch, output_dim].

        """
        # Unsqueeze audio and visual features shape.
        # audio: [batch, audio_dim] -> [batch, 1, audio_dim].
        # video: [batch, video_dim, video_height, video_width]
        #   -> [batch, 1, video_dim, video_height, video_width].
        _audio = audio.unsqueeze(1)
        _video = video.unsqueeze(1)

        # [batch, video_dim]
        v_att = self.att_network(_audio, _video).squeeze(1)

        embed_audio = self.dense_audio(audio)
        embed_video = self.dense_video(v_att)

        return embed_audio, embed_video
