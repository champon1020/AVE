"""Main routine for distance learning

"""

import argparse

from scmm_dataset import SCMMDataset
from scmm_model import AVDLN
from scmm_train import Training
from util import extract_feature, parse_yaml


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.config_path)["scmm"]
    train_config = config["train"]
    model_config = config["model"]

    # If audio and visual features has not been extracted,
    # extract them from video file, and save them.
    if args.extract_features is True:
        extract_feature(args.ave_root)

    model = AVDLN(
        128, 512, model_config["fc_hidden_size"], model_config["fc_output_size"]
    )

    train_ds = SCMMDataset(
        args.ave_root,
        args.train_annot,
        args.features_path,
        train_config["batch_size"],
        model_config["target_size"],
    )

    valid_ds = SCMMDataset(
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
    )

    training.train()


def parse_args() -> argparse.ArgumentParser:
    """Parse CLI arguments

    Returns:
        argparse.ArgumentParser: argument parser.

    """
    parser = argparse.ArgumentParser(description="Audio Visual Distance Learning")

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
        required=False,
    )
    parser.add_argument(
        "--extract-features", help="flag, extract features or not", action="store_true"
    )

    args = parser.parse_args()
    return args
