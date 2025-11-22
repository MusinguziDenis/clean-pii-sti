"""Train YOLO model on the dataset."""

from pathlib import Path

import torch
import yaml
from torch import nn
from ultralytics import YOLO


def get_trained_yolo_models(
        config_files: list[str],
        dataset_path: str,
        device: str = "cuda",
        ) -> list[nn.Module]:
    """Train YOLO models using the specified configuration files and dataset.

    Args:
        config_files (List[str]): List of configuration files
        dataset_path (str): Path to the dataset
        device (str): Device to run the model on

    Returns:
        List[nn.Module]: List of trained YOLO models

    """
    trained_yolo_models = []
    for config_file in config_files:
        file = Path(config_file)
        with file.open("r") as f:
            config_dict = dict(yaml.safe_load(f))

        model = YOLO(config_dict.pop("model"))

        model.train(data=dataset_path, device=device, **config_dict)

        trained_yolo_models.append(model)

    return trained_yolo_models


if __name__ == "__main__":
    config_files = ["yolo11m_config.yaml"]
    dataset_path = "yolo2/dataset.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_models = get_trained_yolo_models(
        config_files,
        dataset_path,
        device)
