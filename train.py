"""Train module for audio visual event localization

This module provides training or validation functions.

"""
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from datasets import AVELDataset
from model import DMRFE
from util import compute_accuracy, parse_yaml


class Training:
    """Training class

    Attributes:
        model (DMRFE): audio visual event localization model.
        train_ds (AVE_Dataset): dataset for training.
        valid_ds (AVE_Dataset): dataset for validation.
        batch_size (int): batch size.
        epoch (int): epoch size.
        learning_rate (int): learning rate.
        valid_span (int): validation span.
        save_span (int): saving model span.
        save_dir (str): directory path to save model.

        optimizer: learning optimizer.
        loss_func: learnig loss function.
        scheduler: learning rate scheduler.

    """

    def __init__(
        self,
        train_ds: AVELDataset,
        valid_ds: AVELDataset,
        model: DMRFE,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
    ):
        self.train_loader = data.DataLoader(train_ds, batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(valid_ds, batch_size, shuffle=True)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.margin_loss = nn.MultiLabelSoftMarginLoss()

    def train(self):
        """
        Training process.

        """
        for epoch in range(self.epochs):
            print("Epoch {} / {}".format(epoch + 1, self.epochs))

            train_loss, train_accuracy = self.train_epoch()
            train_loss = train_loss.item()
            train_accuracy = train_accuracy.item()

            valid_loss, valid_accuracy = self.validate()
            valid_loss = valid_loss.item()
            valid_accuracy = valid_accuracy.item()

            print("[TRAIN] Loss: {:.5}, Acc: {:.5}".format(train_loss, train_accuracy))
            print("[VALID] Loss: {:.5}, Acc: {:.5}".format(valid_loss, valid_accuracy))

    def train_epoch(self) -> Tuple[float, float]:
        """
        Training process executed by epoch.

        """
        self.model.train()
        itr = 0
        batch_loss = 0
        batch_accuracy = 0
        for batch in self.train_loader:
            audio = batch["audio"].cuda()
            visual = batch["visual"].cuda()
            label = batch["label"]["onehot_label"].cuda()

            self.optimizer.zero_grad()
            pred = self.model(audio, visual)

            loss = self.margin_loss(pred, label)
            batch_loss += loss.cpu()
            loss.backward()
            self.optimizer.step()
            batch_accuracy += compute_accuracy(pred, label)
            itr += 1

        loss = batch_loss / float(itr)
        accuracy = batch_accuracy / float(itr)

        return loss, accuracy

    def validate(self):
        """
        Validation process.

        """
        self.model.eval()
        itr = 0
        batch_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                audio = batch["audio"].cuda()
                visual = batch["visual"].cuda()
                label = batch["label"]["onehot_label"].cuda()

                pred = self.model(audio, visual)

                loss = self.margin_loss(pred, label)
                batch_loss += loss.cpu()
                batch_accuracy += compute_accuracy(pred, label)
                itr += 1

        loss = batch_loss / float(itr)
        accuracy = batch_accuracy / float(itr)

        return loss, accuracy


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
        "--train-path",
        help="annotation file path for training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--valid-path",
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

    args = parser.parse_args()
    return args


def main():
    """Main process"""
    args = parse_args()
    config = parse_yaml(args.config_path)

    model = DMRFE(
        128,
        512,
        7 * 7,
        config["att_embed_dim"],
        config["lstm_hidden_dim"],
        config["lstm_num_layers"],
        config["target_size"],
    )

    # AVE training dataset.
    train_ds = AVELDataset(
        args.train_path,
        config["target_size"],
        args.features_path,
        config["n_frames"],
    )

    # AVE validation dataset.
    valid_ds = AVELDataset(
        args.valid_path,
        config["target_size"],
        args.features_path,
        config["n_frames"],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    training = Training(
        train_ds,
        valid_ds,
        model,
        optimizer,
        config["batch_size"],
        config["epochs"],
    )

    training.train()


if __name__ == "__main__":
    main()
