from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn


@dataclass
class LPEConfig:
    """A configuration for a LinearPresuppositionExtractor."""

    # The name of the base model this lens was tuned for.
    base_model_name_or_path: str
    # The
    train_data_path: str
    # The hidden size of the base model.
    d_model: int
    # The number of layers in the base model.
    # num_hidden_layers: int
    # whether to use a bias in the linear translators.
    bias: bool = True
    # The name of the lens type.
    # lens_type: str = "linear_tuned_lens"


class LinearPresuppositionExtractor(nn.Module):
    """A linear presupposition extractor."""

    config: LPEConfig
    linear: nn.Linear

    def __init__(self, config: LPEConfig):
        super(LinearPresuppositionExtractor, self).__init__()
        self.config = config

        self.linear = nn.Linear(
            config.d_model, config.d_model, bias=config.bias
        )
        # nn.init.xavier_normal_(self.linear.weight)
        # nn.init.xavier_normal_(self.linear.bias)

        # self.layer_translators = nn.ModuleList(
        #     [deepcopy(self.linear) for _ in range(5)]
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
        # for layer in self.layer_translators:
        #     x = layer(x)
        # return x

    def __str__(self) -> str:
        return f"LinearPresuppositionExtractor({self.config})"
