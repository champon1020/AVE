"""Train module for audio visual event localization

This module provides training or validation functions.

"""
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from avel_dataset import AVELDataset
from avel_model import DMRFE


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
        model: DMRFE,
        train_ds: AVELDataset,
        valid_ds: AVELDataset,
        batch_size: int,
        epoch: int,
        learning_rate: int,
        valid_span: int,
        save_span: int,
        save_dir: str,
    ):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.frame_num = 10
        self.valid_span = valid_span
        self.save_span = save_span
        self.save_dir = save_dir

        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=25000, gamma=0.1)

    def train(self):
        """Training function"""
        train_loader = DataLoader(self.train_ds, self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_ds, self.batch_size, shuffle=True)

        iterbatch_num = (len(self.train_ds) + 1) // self.batch_size

        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        result_fig = plt.figure()
        loss_ax = result_fig.add_subplot(211)
        acc_ax = result_fig.add_subplot(212)

        for ep in range(self.epoch):
            print("Epoch {0} / {1}".format(ep + 1, self.epoch))

            self.model.train()
            batch_loss = torch.as_tensor(0.0).cuda()
            batch_acc = torch.as_tensor(0.0).cuda()
            for batch in train_loader:
                audio = batch["audio"].cuda()
                video = batch["video"].cuda()
                label = batch["label"]["label"].cuda()

                # Get prediction.
                pred = self.model(audio, video)
                is_correct = torch.argmax(pred, axis=-1) == label

                # Calculate training accuracy.
                acc = torch.sum(is_correct) / float(self.batch_size * self.frame_num)
                batch_acc += acc

                # Convert output
                # from [batch, frame_num, target_size] to [batch, target_size, frame_num]
                # and calculate training loss.
                loss = self.loss_func(pred.permute(0, 2, 1), label)
                loss.backward()

                # Optimize.
                self.optimizer.step()
                self.scheduler.step()
                batch_loss += loss

            # Calculate average loss and accuracy over batch iteration
            # and append them to each lists.
            loss = batch_loss.cpu().detach().numpy() / float(iterbatch_num)
            acc = batch_acc.cpu().detach().numpy() / float(iterbatch_num)
            train_loss.append(loss)
            train_acc.append(acc)

            # Output the result of epoch.
            self._printres("TRAIN", loss, acc)

            # Execute validation.
            if (ep + 1) % self.valid_span == 0:
                self.model.eval()
                self._validate(valid_loader, valid_loss, valid_acc)

            # Save model.
            if (ep + 1) % self.save_span == 0:
                if os.path.exists(self.save_dir) is False:
                    os.mkdir(self.save_dir)
                save_path = os.path.join(self.save_dir, "{0}.pt".format(ep + 1))
                torch.save(self.model, save_path)
                print("Save model as {0}".format(save_path))

        # Plot results.
        self._plot_data(loss_ax, train_loss, self.epoch)
        self._plot_data(loss_ax, valid_loss, self.epoch)
        self._plot_data(acc_ax, train_acc, self.epoch)
        self._plot_data(acc_ax, valid_acc, self.epoch)
        plt.savefig("result.png")

    def _validate(
        self,
        valid_loader: DataLoader,
        valid_loss: List[float],
        valid_acc: List[float],
    ):
        """Validation function

        Args:
            valid_loader (torch.data.DataLoader): validation dataset loader.
            valid_loss (List[float]): validation loss list.
            valid_acc (List[float]): validation accuracy list.

        """
        iterbatch_num = (len(self.valid_ds) + 1) // self.batch_size

        batch_loss = torch.as_tensor(0.0).cuda()
        batch_acc = torch.as_tensor(0.0).cuda()
        for batch in valid_loader:
            audio = batch["audio"].cuda()
            video = batch["video"].cuda()
            label = batch["label"]["label"].cuda()

            pred = self.model(audio, video)
            is_correct = torch.argmax(pred, axis=-1) == label

            # Calculate validation accuracy.
            acc = torch.sum(is_correct) / float(self.batch_size * self.frame_num)
            batch_acc += acc

            # Convert output
            # from [batch, frame_num, target_size] to [batch, target_size, frame_num]
            # and calculate validation loss.
            loss = self.loss_func(pred.permute(0, 2, 1), label)
            batch_loss += loss

        # Calculate average loss and accuracy over batch iteration
        # and append them to each lists.
        loss = batch_loss.cpu().detach().numpy() / float(iterbatch_num)
        acc = batch_acc.cpu().detach().numpy() / float(iterbatch_num)
        valid_loss.append(loss)
        valid_acc.append(acc)

        # Output the result of validation.
        self._printres("VALID", loss, acc)

    @staticmethod
    def _printres(prefix, loss, acc):
        """Print result

        Args:
            loss (float): loss.
            acc (float): accuracy.
            prefix (str): output prefix.

        """
        print("[{0}] loss: {1:.7}, acc: {2:.7}".format(prefix, loss, acc))

    @staticmethod
    def _plot_data(ax, plot_data: List[float], length: int):
        """Plot data to figure

        Args:
            ax (plt.Axes): axes object.
            plot_data (List[float]): data to plot.
            length (int): data length of this figure.

        """
        x = np.linspace(0, length, len(plot_data))
        ax.plot(x, plot_data)
