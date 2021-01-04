"""Model Declaration Module

This module provides model class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DMRFE(nn.Module):
    """DMRFE Model

    This class provides Dual Multimodal Residual Fusion Ensemble Model.

    Attributes:
        audio_dim (int): audio feature dimension.
        video_dim (int): video feature dimension.
        video_size (int): video_height * video_width.
        att_embed_dim (int): embedding dimension of attention network.
        lstm_hidden_dim (int): hidden layer dimension of lstm.
        lstm_num_layers (int): the number of lstm block layers.
        target_size (int):
            the number of targets size. In localization task, this is the number of categories.

        lstm_audio: lstm network for audio feature.
        lstm_video: lstm network for video feature.
        attention_net: audio-guided visual attention network.
        fusion_net: fusion network to joint audio and video features.
        fc: full connected network for event classification.

    """

    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        visual_size: int,
        att_embed_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        target_size: int,
    ):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_dim = lstm_hidden_dim

        self.attention_net = AttentionNet(
            audio_dim,
            visual_dim,
            visual_size,
            att_embed_dim,
        )
        self.lstm_audio = nn.LSTM(
            audio_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_visual = nn.LSTM(
            visual_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fusion_net = FusionNet(
            lstm_hidden_dim * 2,
            lstm_hidden_dim * 2,
            lstm_hidden_dim * 2,
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, target_size)

        self._init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def _init_weights(self):
        """Initialize the weights"""
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, audio: Tensor, visual: Tensor) -> Tensor:
        """
        Args:
            audio (torch.Tensor): Audio feature, [batch, frame_num, audio_dim].
            visual (torch.Tensor):
                Visual feature, [batch, frame_num, visual_height, visual_width, visual_dim].

        Returns:
            (torch.Tensor): Event localization probability, [batch, frame_num, target_size].

        """
        # Get visual attention: [batch, frame_num, visual_dim].
        v_att = self.attention_net(audio, visual)

        # Apply lstm layer.
        h_a, _ = self.lstm_audio(audio)
        h_v, _ = self.lstm_visual(v_att)

        # Joint audio and visual features: [batch, frame_num, hidden_dim*2].
        h_t = self.fusion_net(h_a, h_v)

        # Event localization with softmax.
        h_t = self.fc(h_t)
        out = F.softmax(h_t, dim=-1)

        return out


class AttentionNet(nn.Module):
    """
    This class provides attention network.

    Attributes:
        audio_dim (int): Audio feature dimension.
        visual_dim (int): Visual feature dimension.
        visual_size (int): Visual_height * visual_width.
        embed_dim (int): Embedding dimension.

        affine_audio (int): Dense layer that projects audio feature to embedding space.
        affine_visual (int): Dense layer that projects visual feature to embedding space.
        affine_a (int): Weight parameter.
        affine_v (int): Weight parameter.
        affine_f (int): Weight parameter.

    """

    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        visual_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.affine_audio = nn.Linear(audio_dim, embed_dim)
        self.affine_visual = nn.Linear(visual_dim, embed_dim)
        self.affine_a = nn.Linear(embed_dim, visual_size, bias=False)
        self.affine_v = nn.Linear(embed_dim, visual_size, bias=False)
        self.affine_f = nn.Linear(visual_size, 1, bias=False)

    def forward(self, audio: Tensor, visual: Tensor) -> Tensor:
        """Forward process

        Args:
            audio (torch.Tensor): Audio feature, [batch, n_frames, audio_dim].
            visual (torch.Tensor):
                Visual feature, [batch, n_frames, visual_height, visual_width, visual_dim].

        Returns:
            torch.Tensor: Visual feature attended by audio, [batch, n_frames, visual_dim]

        """
        batch, n_frames, _, _, visual_dim = visual.shape

        # Reshape audio feature: [batch*n_frames, audio_dim].
        audio = audio.view(-1, audio.shape[2])

        # Reshape visual feature: [batch*n_frames, visual_size, visual_dim].
        # where visual_size = visual_height*visual_width
        visual = visual.view(visual.shape[0] * visual.shape[1], -1, visual_dim)

        # Transform audio: [batch*n_frames, embed_dim].
        a_t = F.relu(self.affine_audio(audio))

        # Transform visual: [batch*n_frames, visual_size, embed_dim].
        v_t = F.relu(self.affine_visual(visual))

        # [batch*n_frames, visual_size].
        a_t = self.affine_a(a_t).unsqueeze(2)

        # [batch*n_frames, visual_size, visual_size].
        v_t = self.affine_v(v_t)

        # Add two features: [batch, n_frames, visual_size, 1].
        f_t = self.affine_f(torch.tanh(a_t + v_t))

        # Calculate attention score: [batch*n_frames, visual_size, 1].
        att = F.softmax(f_t, dim=1).view(f_t.shape[0], -1, f_t.shape[1])

        out = torch.bmm(att, visual).view(-1, visual_dim)

        out = out.view(batch, n_frames, -1)

        return out


class FusionNet(nn.Module):
    """Fusion Network (DMRN)

    This class provides fusion network (DMRN).

    Attributes:
        input_size (int): Dense layer input size.
        hidden_size (int): Dense layer hidden size.
        output_size (int): Dense layer output size.

        dense_audio: embedding network for audio.
        dense_visual: embedding network for visual.

    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.dense_audio = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
        self.dense_visual = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, h_audio: Tensor, h_visual: Tensor) -> Tensor:
        """Forward process

        Args:
            h_audio (torch.Tensor): Hidden audio feature, [batch, input_size].
            h_visual (torch.Tensor): Hidden visual feature, [batch, input_size].

        Returns:
            torch.Tensor: Audio visual joint representation, [batch, output_size].

        """
        h_a = self.dense_audio(h_audio)
        h_v = self.dense_visual(h_visual)
        residual_a = h_a
        residual_v = h_v
        merged = torch.mul(h_a + h_v, 0.5)

        h_a = torch.relu(residual_a + merged)
        h_v = torch.relu(residual_v + merged)
        out = torch.mul(h_a + h_v, 0.5)
        return out
