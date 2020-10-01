"""Model Declaration Module

This module provides model class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DMRFE(nn.Module):
    """DMRFE Model

    This class provides Dual Multimodal Residual Fusion Ensemble Model.

    Attributes:
        audio_dim: audio feature dimension.
        video_dim: video feature dimension.
        video_size: video_height * video_width.
        att_embed_dim: embedding dimension of attention network.
        lstm_hidden_dim: hidden layer dimension of lstm.
        lstm_num_layers: the number of lstm block layers.
        target_size:
            the number of targets size. In localization task, this is the number of categories.

    """

    def __init__(
        self,
        audio_dim,
        video_dim,
        video_size,
        att_embed_dim,
        lstm_hidden_dim,
        lstm_num_layers,
        target_size,
    ):
        super().__init__()
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

    def _init_weights(self):
        """Initialize the weights

        """
        nn.init.xavier_uniform(self.fc.weight)

    def forward(self, audio, video):
        """ Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, audio_dim].
            video (torch.Tensor): video feature, [batch, video_dim, video_height, video_width].

        Returns:
            (torch.Tensor): event localization probabilities, [batch, target_size].

        """
        # Get visual attention: [batch, video_dim].
        v_att = self.attention_net(audio, video)

        h_a, _ = self.lstm_audio(audio)
        h_v, _ = self.lstm_video(v_att)

        # Joint audio and visual features: [batch, repl_size].
        h_t = self.fusion_net(h_a, h_v)

        # Event localization with softmax.
        h_t = self.fc(h_t)
        out = F.softmax(h_t, dim=-1)
        return out


class AttentionNet(nn.Module):
    """Attention Network

    This class provides attention network.

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
        """Forward process

        Args:
            audio (torch.Tensor): audio feature, [batch, audio_dim].
            video (torch.Tensor): video feature, [batch, video_dim, video_height, video_width].

        Returns:
            torch.Tensor: video feature attended audio, [batch, video_dim]

        """
        # Reshape video feature:
        #   [batch, video_dim, video_heigth, video_width] -> [batch, video_size, video_dim]
        # where video_size = video_heigth * video_width
        video = video.view(video.shape(0), -1, video.shape(1))

        # Transform audio:
        #   [batch, audio_dim] -> [batch, embed_dim]
        a_t = F.relu(self.affine_audio(audio))

        # Transform video:
        #   [batch, video_size, video_dim] -> [batch, video_size, embed_dim]
        v_t = F.relu(self.affine_video(video))

        # Add two features:
        #   [batch, embed_dim] + [batch, video_size, embed_dim] -> [batch, video_size, video_size]
        f_t = self.affine_a(a_t).unsqueeze(2) + self.affine_v(v_t)

        # Add audio and visual features:
        #   [batch, video_size, video_size] -> [batch, video_size]
        x_t = self.affine_f(F.tanh(f_t)).squeeze(2)

        # Softmax to get attention weight: [batch, 1, video_size]
        w_t = F.softmax(x_t).view(video.shape(0), -1, video.shape(1))

        # Attention map: [batch, video_dim]
        v_att = torch.bmm(w_t, video).squeeze(1)

        return v_att


class FusionNet(nn.Module):
    """Fusion Network (DMRN)

    This class provides fusion network (DMRN).

    Attributes:
        input_size: Dense layer input size.
        hidden_size: Dense layer hidden size.
        output_size: Dense layer output size.

    """

    def __init__(self, input_size, hidden_size, output_size):
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

    def forward(self, h_audio, h_video):
        """Forward process

        Args:
            h_audio (torch.Tensor): hidden audio feature, [batch, input_size].
            h_video (torch.Tensor): hidden video feature, [batch, input_size].

        Returns:
            torch.Tensor: audio video joint representation, [batch, output_size].

        """
        h_t = self.dense_audio(h_audio) + self.dense_video(h_video)
        h_a = F.tanh(h_audio + h_t)
        h_v = F.tanh(h_video + h_t)
        out = torch.mul(h_a + h_v, 0.5)
        return out
