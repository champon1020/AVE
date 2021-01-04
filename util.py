"""Utility module

This module provides some utility functions.

"""
import logging
from typing import Dict

import torch
import yaml
from sklearn.metrics import accuracy_score
from torch import Tensor


def parse_yaml(yaml_path: str) -> Dict[str, str]:
    """Parse yaml configuration file

    Args:
        yaml_path (str): yaml file path.

    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def compute_accuracy(pred: Tensor, target: Tensor):
    """
    Args:
        pred (torch.Tensor): Prediction tensor, [batch, *, target_size].
        target (torch.Tensor): Target tensor, [batch, *, target_size].
    Returns:
        torch.Tensor: Accuracy value.
    """
    pred = torch.argmax(pred, dim=-1).cpu()
    target = torch.argmax(target, dim=-1).cpu()
    accuracy = accuracy_score(pred.view(-1), target.view(-1))
    return accuracy
