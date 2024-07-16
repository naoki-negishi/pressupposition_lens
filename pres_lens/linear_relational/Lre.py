from typing import Any, Literal

import torch
from torch import nn


class Lre(nn.Module):
    """Linear Relational Embedding"""

    relation: str
    subject_layer: int
    object_layer: int
    weight: nn.Parameter
    bias: nn.Parameter
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        weight: torch.Tensor,
        bias: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.relation = relation
        self.subject_layer = subject_layer
        self.object_layer = object_layer
        self.object_aggregation = object_aggregation
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.metadata = metadata

    def forward(
        self,
        subject_acts: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the object activations from the subject activations.
        """
        # print(subject_acts.shape)
        # print(self.weight.shape)
        # print(self.bias.shape)
        lre_score = self.weight @ subject_acts + self.bias
        if normalize:
            lre_score = lre_score / lre_score.norm(dim=0)

        return lre_score


    def __repr__(self) -> str:
        return f"Lre({self.relation}, layers {self.subject_layer} -> {self.object_layer}, {self.object_aggregation})"

