"""Utility module

This module provides some utility functions.

"""
import logging
import os
from typing import Dict

import torch
import yaml

from feature_extractor import FeatureExtractor

logging.basicConfig(format="[AVE '%(levelname)s] %(message)s : %(asctime)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_yaml(yaml_path: str) -> Dict[str, str]:
    """Parse yaml configuration file

    Args:
        yaml_path (str): yaml file path.

    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def extract_feature(ave_root: str):
    """Main routine of extracting features

    Args:
        ave_root (str): AVE dataset root path.

    """
    # Check if the audio features directory is exist.
    if os.path.exists("features/audio") is False:
        os.makedirs("features/audio")
        logger.info("make directory features/audio")

    # Check if the frame features directory is exist.
    if os.path.exists("features/frame") is False:
        os.makedirs("features/frame")
        logger.info("make directory features/frame")

    # Check if the .wav files directory is exist.
    if os.path.exists("waves") is False:
        os.makedirs("waves")
        logger.info("make directory waves")

    logger.info("Extracting features...")
    video_files = os.listdir(ave_root)
    feature_extractor = FeatureExtractor()

    for i, video_name in enumerate(video_files):
        video_path = os.path.join(ave_root, video_name)
        video_id = os.path.splitext(video_name)[0]

        audio_output_name = "features/audio/{0}.pt".format(video_id)
        frame_output_name = "features/frame/{0}.pt".format(video_id)

        if os.path.exists(audio_output_name) and os.path.exists(frame_output_name):
            print("Skip extract {0}.mp4".format(video_id))
            continue

        feature_a, feature_v = feature_extractor.extract(video_path)
        torch.save(feature_a, audio_output_name)
        torch.save(feature_v, frame_output_name)
        print("Save {0}".format(video_id))
        print("Status: {0} / {1}".format(i + 1, len(video_files)))
