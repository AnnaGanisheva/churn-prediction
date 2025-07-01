import os
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from ensure import ensure_annotations

from src.utils.logger import logger


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    with open(path, "r") as f:
        config = ConfigBox(yaml.safe_load(f))
        logger.info(f"yaml file: {path} loaded successfully")
    return config


# @ensure_annotations
# def create_directories(path_to_directories: list, verbose=True):
#     """create list of directories

#     Args:
#         path_to_directories (list): list of path of directories
#         ignore_log (bool, optional): ignore if multiple dirs is to be created.
#         Defaults to False.
#     """
#     for path in path_to_directories:
#         os.makedirs(path, exist_ok=True)
#         if verbose:
#             logger.info(f"created directory at: {path}")


# def save_bin(data, path: Path):
#     """save binary file

#     Args:
#         data (Any): data to be saved as binary
#         path (Path): path to binary file
#     """
#     joblib.dump(value=data, filename=path)
#     logger.info(f"binary file saved at: {path}")


# @ensure_annotations
# def load_bin(path: Path) -> Any:
#     """load binary data

#     Args:
#         path (Path): path to binary file

#     Returns:
#         Any: object stored in the file
#     """
#     data = joblib.load(path)
#     logger.info(f"binary file loaded from: {path}")
#     return data
