"""Feature extractor module

This module provides some functions to extract pretrained audio and visual features.

"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor


class FeatureExtractor:
    """Feature extractor class

    Attributes:
        vgg19_model (torch.nn.Module): vgg19 model.
        avgpool (torch.nn.AdaptiveAvgPool2d): GAP.
        vggish_model (torch.nn.Module): vgg like model to get audio feature.

    """

    def __init__(self):
        self.vgg19_model = torchvision.models.vgg19_bn(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((None, 10))
        self.vggish_model = torch.hub.load("harritaylor/torchvggish", "vggish")

        if torch.cuda.is_available():
            self.vgg19_model.to("cuda")

    def extract(self, video_path: str) -> (Tensor, Tensor):
        """Extract video and audio features

        Args:
            video_path (str): video file relative path.

        Returns:
            torch.Tensor: audio feature.
            torch.Tensor: frame feature.

        """
        wav_path = self._mp4_to_wav(video_path)
        vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        frames = []

        for i in range(vframes.size(0)):
            # Transpose (H, W, C) to (C, H, W).
            frame = vframes[i].permute(2, 0, 1)
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            frame = transform(frame)
            frames.append(frame.numpy())

        # Convert frames to torch.Tensor.
        frames = torch.from_numpy(np.array(frames))

        feature_a = self._extract_audio_feature(wav_path)
        feature_v = self._extract_video_feature(frames)

        return feature_a, feature_v

    def _extract_video_feature(self, frame: Tensor) -> Tensor:
        """Extract video feature

        Args:
            frame (torch.Tensor): frame tensor.

        Returns:
            torch.Tensor: frame embedding.

        """

        if torch.cuda.is_available():
            frame = frame.to("cuda")

        with torch.no_grad():
            embedding = self.vgg19_model(frame)
            # (250, 512, 7, 7) to (512, 7, 7, 250)
            embedding = embedding.permute(1, 2, 3, 0)
            # Execute GAP, (512, 7, 7, 10)
            embedding = self.avgpool(embedding)
            # (512, 7, 7, 10) to (10, 512, 7, 7)
            embedding = embedding.permute(3, 0, 1, 2)

        return embedding

    def _extract_audio_feature(self, wave_path: str) -> Tensor:
        """Extract audio feature

        Args:
            wave_path (str): audio file path with wave format.

        Returns:
            torch.Tensor: audio embedding.

        """
        with torch.no_grad():
            embedding = self.vggish_model.forward(wave_path)

        return embedding

    @staticmethod
    def _mp4_to_wav(video_path: str) -> str:
        """Extract audio file with wav format from mp4 video file

        Args:
            video_path (str): video file relative path.

        Returns:
            str: output audio file path.

        """
        video_path_without_ext = os.path.join(
            "waves", os.path.splitext(os.path.basename(video_path))[0]
        )
        wav_path = "{0}.wav".format(video_path_without_ext)

        if os.path.exists(wav_path) is False:
            command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(
                video_path, wav_path
            )

            if subprocess.call(command, shell=True) != 0:
                print("Failed to execute '{0}'".format(command))
                sys.exit(1)
        else:
            print("Skip {0}".format(wav_path))

        return wav_path
