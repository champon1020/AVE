"""Main routine for audio visual event localization

This module provides main process functions.

"""
import argparse
import logging
import os

import torch

from avel_dataset import AVELDataset
from avel_model import DMRFE
from avel_train import Training
from feature_extractor import FeatureExtractor
from util import parse_yaml

logging.basicConfig(format="[AVE '%(levelname)s] %(message)s : %(asctime)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.yaml_path)
    train_config = config["train"]
    model_config = config["model"]

    # If audio and visual features has not been extracted,
    # extract them from video file, and save them.
    if args.extract_features is True:
        extract_feature_main(args)

    model = DMRFE(
        128,
        512,
        7 * 7,
        model_config["att_embed_dim"],
        model_config["lstm_hidden_dim"],
        model_config["lstm_num_layers"],
        model_config["target_size"],
    )

    # AVE training dataset.
    train_ds = AVELDataset(
        args.ave_root,
        args.train_annot,
        args.features_path,
        train_config["batch_size"],
        model_config["target_size"],
    )

    # AVE validation dataset.
    valid_ds = AVELDataset(
        args.ave_root,
        args.valid_annot,
        args.features_path,
        train_config["batch_size"],
        model_config["target_size"],
    )

    training = Training(
        model,
        train_ds,
        valid_ds,
        train_config["batch_size"],
        train_config["epoch"],
        train_config["learning_rate"],
        train_config["valid_span"],
        train_config["save_span"],
        train_config["save_dir"],
    )

    training.train()


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
        "--train-annot",
        help="annotation file path for training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--valid-annot",
        help="annotation file path for validation",
        type=str,
        required=True,
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
