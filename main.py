"""Main routine

This module provides main process functions.

"""
import argparse
import logging
import os
from typing import Dict

import torch
import yaml

from dataloader import AVEDataset
from feature_extractor import FeatureExtractor
from model import DMRFE
from train import Training

logging.basicConfig(format="[AVE '%(levelname)s] %(message)s : %(asctime)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args() -> argparse.ArgumentParser:
    """Parse CLI arguments

    Returns:
        argparse.ArgumentParser: argument parser.

    """
    parser = argparse.ArgumentParser(description="Audio Visual Event")

    parser.add_argument(
        "--yaml-path", help="configuration file path of yaml", type=str, required=True
    )
    parser.add_argument(
        "--ave-root", help="AVE dataset root path", type=str, required=True
    )
    parser.add_argument(
        "--annot-path", help="annotation file path", type=str, required=True
    )
    parser.add_argument(
        "--features-path",
        default="./features",
        help="features directory path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--extract-features", help="flag, extract features or not", action="store_true"
    )

    args = parser.parse_args()
    return args


def parse_yaml(yaml_path: str) -> Dict[str, str]:
    """Parse yaml configuration file

    Args:
        yaml_path (str): yaml file path.

    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.yaml_path)
    train_config = config["train"]

    # If audio and visual features has not been extracted,
    # extract them from video file, and save them.
    if args.extract_features is True:
        extract_feature_main(args)

    # AVE training dataset.
    train_ds = AVEDataset(
        args.ave_root,
        args.annot_path,
        args.features_path,
        train_config["batch_size"],
        train_config["target_size"],
    )

    # AVE validation dataset.
    valid_ds = AVEDataset(
        args.ave_root,
        args.annot_path,
        args.features_path,
        train_config["batch_size"],
        train_config["target_size"],
    )

    model = DMRFE(
        128,
        512,
        7 * 7,
        train_config["att_embed_dim"],
        train_config["lstm_hidden_dim"],
        train_config["lstm_num_layers"],
        train_config["target_size"],
    )

    training = Training(
        model,
        train_ds,
        valid_ds,
        train_config["batch_size"],
        train_config["epoch"],
        train_config["learning_rate"],
        train_config["valid_span"],
    )

    training.train()


def extract_feature_main(args):
    """Main routine of extracting features"""
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
    video_files = os.listdir(args.ave_root)
    feature_extractor = FeatureExtractor()

    for i, video_name in enumerate(video_files):
        video_path = os.path.join(args.ave_root, video_name)
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


if __name__ == "__main__":
    main()
