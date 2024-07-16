import random
from typing import Any

import torch


def sampling_inst(dataset: Any, instance_num: int) -> list[Any]:
    sampled_instances = random.sample(dataset, instance_num)
    return sampled_instances


def expectation(w: list[torch.Tensor]) -> torch.Tensor:
    return torch.mean(w)


def save_img(visualized_img: Any, path: str) -> None:
    visualized_img.save(path)
