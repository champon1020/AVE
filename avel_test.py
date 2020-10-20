"""Test functions

This module provides some functions for testing model.

"""
import argparse

import torch
from torch.utils.data import DataLoader

from avel_dataset import AVELDataset
from avel_model import DMRFE
from util import parse_yaml


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.yaml_path)
    test_config = config["test"]
    model_config = config["model"]

    # Load model.
    model = torch.load(args.ckpt_path)

    # If test mode, execute test and finish the main process.
    test_ds = AVELDataset(
        args.ave_root,
        args.test_annot,
        args.features_path,
        test_config["batch_size"],
        model_config["target_size"],
    )

    test(model, test_ds)


def test(model: DMRFE, test_ds: AVELDataset):
    """Test function

    Args:
        model (DMRFE): model class.
        test_ds (AVELDataset): AVE dataset for testing.

    """
    batch_size = 1
    frame_num = 10
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)
    iterbatch_num = (len(test_ds) + 1) // batch_size

    model.eval()
    batch_acc = torch.as_tensor(0.0).cuda()
    for batch in test_loader:
        audio = batch["audio"].cuda()
        video = batch["video"].cuda()
        label = batch["label"]["label"].cuda()

        pred = model(audio, video)
        is_correct = torch.argmax(pred, axis=-1) == label

        # Calculate test accuracy.
        acc = torch.sum(is_correct) / float(batch_size * frame_num)
        batch_acc += acc

    acc = batch_acc.cpu().detach().numpy() / float(iterbatch_num)
    print("Test Average Accuracy: {0:.5}".format(acc))


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
        "--test-annot",
        help="annotation file path for training",
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
        "--ckpt-path",
        help="checkpoint file path",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
