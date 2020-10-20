"""Train module for audio visual distance learnint

This module provides training or validation functions.

"""
import os
from typing import List

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from cmm_dataset import CMMDataset
from cmm_model import AVDLN


class Training:
    """Training class for audio visual distance learning

    Attributes:
        model (AVDLN): audio visual distance learning model.
        train_ds (CMMDataset): training dataset.
        valid_ds (CMMDataset): validation dataset.
        batch_size (int): batch size.
        epoch (int): epoch size.
        learning_rate (float): learning rate.
        loss_margin (float): contrastive loss margin.
        valid_span (int): validation span.
        save_span (int): saving model span.
        save_dir (str): directory path to save model.

        optimizer: learning optimizer.
        scheduler: learning rate scheduler.
    """

    def __init__(
        self,
        model: AVDLN,
        train_ds: CMMDataset,
        valid_ds: CMMDataset,
        batch_size: int,
        epoch: int,
        learning_rate: float,
        loss_margin: float,
        valid_span: int,
        save_span: int,
        save_dir: str,
    ):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss_margin = loss_margin
        self.valid_span = valid_span
        self.save_span = save_span
        self.save_dir = save_dir

        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=25000, gamma=0.1)

    @staticmethod
    def _euclidean_distance(vec1: Tensor, vec2: Tensor) -> Tensor:
        """Calculate euclidean distance between two vectors.

        Args:
            vec1 (torch.Tensor): tensor.
            vec2 (torch.Tensor): tensor.

        Returns:
            torch.Tensor: euclidean distance with tensor.

        """
        return torch.cdist(vec1, vec2)

    def _contrastive_loss(self, audio: Tensor, video: Tensor, y: Tensor) -> Tensor:
        """Contrastive loss

        Args:
            audio (torch.Tensor): audio embedding tensor.
            video (torch.Tensor): video embedding tensor.
            y (torch.Tensor<int>): wheather audio and video is corresponding or not.

        Returns:
            torch.Tensor<float>: loss.

        """
        dist = self._euclidean_distance(audio, video)

        loss_pos = y * (dist ** 2)
        loss_neg = (1 - y) * (
            torch.max(torch.zeros_like(dist), self.loss_margin - dist) ** 2
        )

        return loss_pos + loss_neg

    def train(self):
        """Traiinig function"""
        train_loader = DataLoader(self.train_ds, self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_ds, self.batch_size, shuffle=True)

        iterbatch_num = (len(self.train_ds) + 1) // self.batch_size

        train_loss = []
        valid_acc = []

        for ep in range(self.epoch):
            print("Epoch {0} / {1}".format(ep + 1, self.epoch))

            self.model.train()
            batch_loss = torch.as_tensor(0.0).cuda()
            for batch in train_loader:
                audio = batch["audio"].cuda()
                video = batch["video"].cuda()
                label = batch["label"].cuda()

                # Audio and video embedding.
                embed_audio, embed_video = self.model(audio, video)

                # Calculate loss.
                loss = self._contrastive_loss(embed_audio, embed_video, label)
                loss.backward()
                batch_loss += loss

                # Optimize.
                self.optimizer.step()
                self.scheduler.step()

            loss = batch_loss.cpu().detach().numpy() / float(iterbatch_num)
            train_loss.append(loss)

            print("[TRAIN] Loss: {0:.7}".format(loss))

            # Validation.
            if (ep + 1) % self.valid_span == 0:
                self.model.eval()
                self._validation(valid_loader, valid_acc)

            # Save model.
            if (ep + 1) % self.save_span == 0:
                if os.path.exists(self.save_dir) is False:
                    os.mkdir(self.save_dir)
                save_path = os.path.join(self.save_dir, "{0}.pt".format(ep + 1))
                torch.save(self.model, save_path)
                print("Save model as {0}".format(save_path))

    def _validation(self, valid_loader: DataLoader, valid_acc: List[float]):
        """Validation function

        Args:
            valid_loader (torhc.DataLoader): validation dataset loader.
            valid_acc (List[float]): validation accuracy list.

        """
        iterbatch_num = (len(self.valid_ds) + 1) // self.batch_size
        return
