"""Main routine for audio visual event localization

This module provides main process functions.

"""
import argparse

from avel_dataset import AVELDataset
from avel_model import DMRFE
from avel_train import Training
from util import extract_feature, parse_yaml


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.config_path)["avel"]
    train_config = config["train"]
    model_config = config["model"]

    # If audio and visual features has not been extracted,
    # extract them from video file, and save them.
    if args.extract_features is True:
        extract_feature(args.ave_root)

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
    parser = argparse.ArgumentParser(description="Audio Visual Event Localization")

    parser.add_argument(
        "--config-path", help="configuration file path", type=str, required=True
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
        required=True,
    )
    parser.add_argument(
        "--extract-features",
        help="flag, extract features or not",
        action="store_true",
        required=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
