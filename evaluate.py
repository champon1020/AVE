"""Test functions

This module provides some functions for testing model.

"""
import argparse

import torch
import torch.utils.data as data

from datasets import AVELDataset
from model import DMRFE
from util import compute_accuracy, parse_yaml


class Evaluation:
    """
    This class provides evaluation functions.

    Attributes:
        test_loader (Dataset): Testing dataset loader.
        model (Model): Model class.

    """

    def __init__(self, test_ds: AVELDataset, model: DMRFE):
        self.test_loader = data.DataLoader(test_ds, 1, shuffle=False)
        self.model = model

    def evaluate(self) -> float:
        """
        Evaluation process

        """
        self.model.eval()
        itr = 0
        batch_accuracy = 0
        for batch in self.test_loader:
            audio = batch["audio"].cuda()
            visual = batch["visual"].cuda()
            label = batch["label"]["onehot_label"].cuda()
            pred = self.model(audio, visual)

            batch_accuracy += compute_accuracy(pred, label)

            itr += 1

        accuracy = batch_accuracy / float(itr)

        print(accuracy)


def parse_args() -> argparse.ArgumentParser:
    """Parse CLI arguments

    Returns:
        argparse.ArgumentParser: argument parser.

    """
    parser = argparse.ArgumentParser(description="Audio Visual Event")

    parser.add_argument(
        "--config-path", help="configuration file path", type=str, required=True
    )
    parser.add_argument(
        "--test-path",
        help="annotation file path for training",
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
        "--ckpt-path",
        help="checkpoint file path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.config_path)

    # Load model.
    model = torch.load(args.ckpt_path)

    # If test mode, execute test and finish the main process.
    test_ds = AVELDataset(
        args.test_path,
        config["target_size"],
        args.features_path,
        config["n_frames"],
    )

    evaluation = Evaluation(test_ds, model)
    evaluation.evaluate()


if __name__ == "__main__":
    main()
