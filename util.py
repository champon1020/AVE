"""Utility module

This module provides some utility functions.

"""
from typing import Dict

import yaml


def parse_yaml(yaml_path: str) -> Dict[str, str]:
    """Parse yaml configuration file

    Args:
        yaml_path (str): yaml file path.

    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data
