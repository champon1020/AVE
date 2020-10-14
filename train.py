"""Train module

This module provides training or validation functions.

"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataloader import AVEDataset
from model import DMRFE


class Training:
    """Training class

    Attributes:
        model (DMRFE): audio visual event localization model.
        train_ds (AVE_Dataset): dataset for training.
        valid_ds (AVE_Dataset): dataset for validation.
        batch_size (int): batch size.
        epoch (int): epoch size.
        learning_rate (int): learning rate.
        optimizer: learning optimizer.
        loss_func: learnig loss function.

    """

    def __init__(
        self,
        model: DMRFE,
        train_ds: AVEDataset,
        valid_ds: AVEDataset,
        batch_size: int,
        epoch: int,
        learning_rate: int,
        valid_span: int,
    ):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.frame_num = 10
        self.valid_span = valid_span

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        """Training function"""
        train_loader = data.DataLoader(self.train_ds, self.batch_size, shuffle=True)
        iterbatch_num = (len(self.train_ds) + 1) // self.batch_size
        valid_loader = data.DataLoader(self.valid_ds, self.batch_size, shuffle=True)

        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        result_fig = plt.figure()
        loss_ax = result_fig.add_subplot(111)
        acc_ax = result_fig.add_subplot(121)

        for ep in range(self.epoch):
            batch_loss = torch.as_tensor(0.0).cuda()
            batch_acc = torch.as_tensor(0.0).cuda()
            for batch in train_loader:
                audio = batch["audio"].cuda()
                video = batch["video"].cuda()
                label = batch["label"]["label"].cuda()

                pred = self.model(audio, video)
                is_correct = torch.argmax(pred, axis=-1) == label

                acc = torch.sum(is_correct) / float(self.batch_size * self.frame_num)
                batch_acc += acc

                # Convert output
                # from [batch, frame_num, target_size] to [batch, target_size, frame_num]
                # and apply loss function.
                loss = self.loss_func(pred.permute(0, 2, 1), label)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss

            # Calcurate average loss and accuracy over batch iteration
            # and append them to each lists.
            train_loss.append(batch_loss.cpu().detach().numpy() / float(iterbatch_num))
            train_acc.append(batch_acc.cpu().detach().numpy() / float(iterbatch_num))

            # Output the result of epoch.
            self._printres("TRAIN", train_loss[-1], train_acc[-1])

            # Execute validation.
            if (ep + 1) % self.valid_span == 0:
                self._validate(valid_loader, valid_loss, valid_acc)

        self._plot_data(loss_ax, train_loss, self.epoch)
        self._plot_data(loss_ax, valid_loss, self.epoch)
        self._plot_data(acc_ax, train_acc, self.epoch)
        self._plot_data(acc_ax, valid_acc, self.epoch)
        plt.savefig("result.png")

    def _validate(
        self,
        valid_loader: data.DataLoader,
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

            # Convert output
            # from [batch, frame_num, target_size] to [batch, target_size, frame_num]
            pred = self.model(audio, video)
            is_correct = torch.argmax(pred, axis=-1) == label

            acc = torch.sum(is_correct) / float(self.batch_size * self.frame_num)
            batch_acc += acc

            loss = self.loss_func(pred.permute(0, 2, 1), label)
            batch_loss += loss

        # Calcurate average loss and accuracy over batch iteration
        # and append them to each lists.
        valid_loss.append(loss.cpu().detach().numpy() / float(iterbatch_num))
        valid_acc.append(acc.cpu().detach().numpy() / float(iterbatch_num))

        # Output the result of validation.
        self._printres("VALID", valid_loss[-1], valid_acc[-1])

    @staticmethod
    def _printres(prefix, loss, acc):
        """Print result

        Args:
            loss (float): loss.
            acc (float): accuracy.
            prefix (str): output prefix.

        """
        print("[{0}] loss: {1}, acc: {2}".format(prefix, loss, acc))

    @staticmethod
    def _plot_data(ax, data, length):
        """Plot data to figure

        Args:
            ax (plt.Axes): axes object.
            data (List[float]): data to plot.
            length (int): data length of this figure.

        """
        x = np.linspace(0, 1, length)
        ax.plot(x, data)
