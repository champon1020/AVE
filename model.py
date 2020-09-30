"""Model Declaration Module

This module provides module class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):
    """Attention Network

    This class provides attention network for DMRFE model.

    Attributes:
        affine_audio: Dense layer that projects audio feature to embedding space.
        affine_video: Dense layer that projects video feature to embedding space.
        affine_a: Weight parameter.
        affine_v: Weight parameter.
        affine_f: Weight parameter.

    """

    def __init__(self, audio_dim, video_dim, video_size, embed_dim):
        super().__init__()
        self.affine_audio = nn.Linear(audio_dim, embed_dim)
        self.affine_video = nn.Linear(video_dim, embed_dim)
        self.affine_a = nn.Linear(embed_dim, video_size, bias=False)
        self.affine_v = nn.Linear(embed_dim, video_size, bias=False)
        self.affine_f = nn.Linear(video_size, 1, bias=False)

    def forward(self, audio, video):
        """
        Args:
            audio (torch.Tensor): audio feature, [batch, audio_dim].
            video (torch.Tensor):
                video feature, [batch, video_dim, video_height, video_width].

        Returns:
            torch.Tensor: video feature attended audio.

        """
        # Reshape video feature: [batch, video_dim, video_heigth, video_width]
        #   -> [batch, video_size, video_dim]
        # where video_size = video_heigth * video_width
        video = video.view(video.shape(0), -1, video.shape(1))

        # Transform audio: [batch, audio_dim] -> [batch, embed_dim]
        a_t = F.relu(self.affine_audio(audio))

        # Transform video: [batch, video_size, video_dim]
        #   -> [batch, video_size, embed_dim]
        v_t = F.relu(self.affine_video(video))

        # Add two features: [batch, embed_dim] + [batch, video_size, embed_dim]
        #   -> [batch, video_size, video_size]
        f_t = self.affine_a(a_t).unsqueeze(2) + self.affine_v(v_t)

        # Add audio and visual features: [batch, video_size, video_size]
        #   -> [batch, video_size]
        x_t = self.affine_f(F.tanh(f_t)).squeeze(2)

        # Softmax to get attention weight: [batch, 1, video_size]
        w_t = F.softmax(x_t).view(video.shape(0), -1, video.shape(1))

        # Attention map: [batch, video_dim]
        v_att = torch.bmm(w_t, video).squeeze(1)

        return v_att
