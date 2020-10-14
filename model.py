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

    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        video_size: int,
        att_embed_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        target_size: int,
    ):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_dim = lstm_hidden_dim
    
        self.attention_net = AttentionNet(
            audio_dim, video_dim, video_size, att_embed_dim
        )
        self.fusion_net = FusionNet(
            lstm_hidden_dim * 2, lstm_hidden_dim * 2, lstm_hidden_dim * 2
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, target_size)

        self.lstm_audio = nn.LSTM(
            audio_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_video = nn.LSTM(
            video_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self._init_weights()

        if torch.cuda.is_available():
            self.cuda()

    def _init_weights(self):
        """Initialize the weights"""
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, audio: Tensor, video: Tensor) -> Tensor:
        """Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, frame_num, audio_dim].
            video (torch.Tensor):
                video feature, [batch, frame_num, video_dim, video_height, video_width].

        Returns:
            (torch.Tensor): event localization probabilities, [batch, frame_num, target_size].

        """
        # Get visual attention: [batch, frame_num, video_dim].
        v_att = self.attention_net(audio, video)

        # Apply lstm layer.
        h_a, _ = self.lstm_audio(audio)
        h_v, _ = self.lstm_video(v_att)

        # Joint audio and visual features: [batch, frame_num, hidden_dim*2].
        h_t = self.fusion_net(h_a, h_v)

        # Event localization with softmax.
        h_t = self.fc(h_t)
        out = F.softmax(h_t, dim=-1)

        return out


class AttentionNet(nn.Module):
    """Attention Network

    This class provides attention network.

    Attributes:
        affine_audio (int): Dense layer that projects audio feature to embedding space.
        affine_video (int): Dense layer that projects video feature to embedding space.
        affine_a (int): Weight parameter.
        affine_v (int): Weight parameter.
        affine_f (int): Weight parameter.

    """

    def __init__(self, audio_dim: int, video_dim: int, video_size: int, embed_dim: int):
        super().__init__()
        self.affine_audio = nn.Linear(audio_dim, embed_dim)
        self.affine_video = nn.Linear(video_dim, embed_dim)
        self.affine_a = nn.Linear(embed_dim, video_size, bias=False)
        self.affine_v = nn.Linear(embed_dim, video_size, bias=False)
        self.affine_f = nn.Linear(embed_dim, 1, bias=False)


    def forward(self, audio: Tensor, video: Tensor) -> Tensor:
        """Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, frame_num, audio_dim].
            video (torch.Tensor):
                video feature, [batch, frame_num, video_dim, video_height, video_width].

        Returns:
            torch.Tensor: video feature attended audio, [batch, video_dim]

        """
        # Reshape video feature:
        #   [batch, frame_num, video_dim, video_heigth, video_width]
        #   -> [batch, frame_num, video_size, video_dim]
        # where video_size = video_heigth * video_width
        video = video.view(video.shape[0], video.shape[1], -1, video.shape[2])

        # Transform audio:
        #   [batch, frame_num, audio_dim] -> [batch, frame_num, embed_dim]
        a_t = F.relu(self.affine_audio(audio))

        # Transform video:
        #   [batch, frame_num, video_size, video_dim] -> [batch, frame_num, video_size, embed_dim]
        v_t = F.relu(self.affine_video(video))

        # Add two features:
        #   [batch, frame_num, embed_dim] + [batch, frame_num, video_size, embed_dim]
        #   -> [batch, frame_num, video_size, embed_dim]
        f_t = self.affine_a(a_t).unsqueeze(2) + self.affine_v(v_t)

        # Add audio and visual features:
        #   [batch, frame_num, video_size, embed_dim] -> [batch*frame_num, 1, video_size]
        x_t = self.affine_f(torch.tanh(f_t)).view(-1, 1, f_t.shape[2])

        # Softmax to get attention weight: [batch*frame_num, 1, video_size]
        w_t = F.softmax(x_t, dim=2)

        # Attention map: [batch*frame_num, 1, video_dim]
        v_att = torch.bmm(
            w_t,
            video.view(video.shape[0] * video.shape[1], video.shape[2], video.shape[3]),
        )

        # Convert shape to [batch, frame_num, video_dim]
        v_att = v_att.view(video.shape[0], video.shape[1], video.shape[3])

        return v_att


class FusionNet(nn.Module):
    """Fusion Network (DMRN)

    This class provides fusion network (DMRN).

    Attributes:
        input_size (int): Dense layer input size.
        hidden_size (int): Dense layer hidden size.
        output_size (int): Dense layer output size.

    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.dense_audio = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
        self.dense_video = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, h_audio: Tensor, h_video: Tensor) -> Tensor:
        """Forward process

        Args:
            h_audio (torch.Tensor): hidden audio feature, [batch, input_size].
            h_video (torch.Tensor): hidden video feature, [batch, input_size].

        Returns:
            torch.Tensor: audio video joint representation, [batch, output_size].

        """
        h_t = self.dense_audio(h_audio) + self.dense_video(h_video)
        h_a = torch.tanh(h_audio + h_t)
        h_v = torch.tanh(h_video + h_t)
        out = torch.mul(h_a + h_v, 0.5)
        return out
