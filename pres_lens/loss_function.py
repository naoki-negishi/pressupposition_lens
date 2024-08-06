# Standard Library
from typing import Callable

# Third Party Library
import torch
from torch import nn


class LossFunctionForLPE:
    loss_function: Callable[[torch.Tensor, torch.Tensor], float]

    def __init__(self, loss_function: str):
        if loss_function == "CE":
            loss = nn.CrossEntropyLoss()
            self.loss_function = self.ce(loss)
        # elif loss_function == "contrastive":
        #     self.loss_function = contrastive
        # elif loss_function == "triplet":
        #     self.loss_function = triplet
        # elif loss_function == "batch_all_triplet":
            # self.loss_function = batch_all_triplet
        # elif loss_function == "KL":
        #     self.loss_function = kl
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")

    def __call__(self, y: torch.Tensor, z: torch.Tensor) -> float:
        assert y.shape == z.shape
        loss = self.loss_function(y, z)

        return loss

    def ce(self, ce_loss_func):
        # loss = th.nn.functional.cross_entropy(
        #     logits.flatten(0, -2), labels.flatten()
        # )
        return ce_loss_func

    def contrastive(y: torch.Tensor, z: torch.Tensor) -> float:
        pass

    def triplet(y: torch.Tensor, z: torch.Tensor) -> float:
        pass

    def batch_all_triplet(y: torch.Tensor, z: torch.Tensor) -> float:
        pass

    # def kl(y: torch.Tensor, z: torch.Tensor) -> float:
    #     labels = final_logits.float().log_softmax(dim=-1)
    #     loss = th.sum(
    #         labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
    #     ).mean()
    #
    #     return loss
