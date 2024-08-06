from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn


@dataclass
class LPEConfig:
    """A configuration for a LinearPresuppositionExtractor."""

    # The name of the base model this lens was tuned for.
    base_model_name_or_path: str
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
    extractor: nn.Linear

    def __init__(self, config: LPEConfig):
        super(LinearPresuppositionExtractor, self).__init__()
        self.config = config

        # TODO: dtype=torch.float32?
        self.extractor = nn.Linear(
            config.d_model, config.d_model, bias=config.bias
        )
        self.extractor.weight.data.zero_()
        self.extractor.bias.data.zero_()

        # Don't include the final layer since it does not need a extractor
        # self.layer_translators = nn.ModuleList(
        #     [deepcopy(extractor) for _ in range(self.config.num_hidden_layers)]
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)
