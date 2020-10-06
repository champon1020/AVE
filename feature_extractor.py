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


def extract_features(video_path):
    """Extract video and audio features

    Args:
        video_path (str): video file relative path.

    Returns:
        torch.Tensor: audio feature.
        torch.Tensor: frame feature.

    """
    wav_path = _mp4_to_wav(video_path)
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

    feature_a = _extract_audio_feature(wav_path)
    feature_v = _extract_video_feature(frames)

    return feature_a, feature_v


def _extract_video_feature(frame):
    """Extract video feature

    Args:
        frame (torch.Tensor): frame tensor.

    Returns:
        torch.Tensor: frame embedding.

    """
    vgg19_model = torchvision.models.vgg19_bn(pretrained=True).features
    avgpool = nn.AdaptiveAvgPool2d((None, 10))

    if torch.cuda.is_available():
        frame = frame.to("cuda")
        vgg19_model.to("cuda")

    with torch.no_grad():
        embedding = vgg19_model(frame)
        # (250, 512, 7, 7) to (512, 7, 7, 250)
        embedding = embedding.permute(1, 2, 3, 0)
        # Execute GAP, (512, 7, 7, 10)
        embedding = avgpool(embedding)
        # (512, 7, 7, 10) to (10, 512, 7, 7)
        embedding = embedding.permute(3, 0, 1, 2)

    return embedding


def _extract_audio_feature(wave_path):
    """Extract audio feature

    Args:
        wave_path (str): audio file path with wave format.

    Returns:
        torch.Tensor: audio embedding.

    """
    vggish_model = torch.hub.load("harritaylor/torchvggish", "vggish")

    with torch.no_grad():
        embedding = vggish_model.forward(wave_path)

    return embedding


def _mp4_to_wav(video_path):
    """Extract audio file with wav format from mp4 video file

    Args:
        video_path (str): video file relative path.

    Returns:
        str: output audio file path.

    """
    video_path_without_ext = os.path.join(
        os.path.dirname(video_path), os.path.splitext(os.path.basename(video_path))[0]
    )
    wav_path = "{0}.wav".format(video_path_without_ext)

    if os.path.exists(wav_path) is False:
        command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(
            video_path, wav_path
        )

        if subprocess.call(command, shell=True) != 0:
            print("Failed to execute '{0}'".format(command))
            sys.exit(1)

    return wav_path
