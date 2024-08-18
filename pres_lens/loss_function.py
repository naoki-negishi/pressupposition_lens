# Standard Library
from abc import ABC, abstractmethod

# Third Party Library
import torch
from torch import nn


class LossFunction(ABC):
    def __init__(self):
        pass

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor
    ) -> float:
        return self.compute_loss(inputs, targets, labels)

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> float:
        pass


class ContrastiveLoss(LossFunction):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.0
        self.cos_emb_loss = nn.CosineEmbeddingLoss(
            margin=self.margin, reduction="mean"
        )

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor
    ) -> float:
        # inputs.shape == targets.shape == [batch_size, embedding_dim]
        # label.shape == [batch_size]
        assert inputs.shape == targets.shape
        loss = self.cos_emb_loss(inputs, targets, labels)

        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - nn.CosineSimilarity()(x, y),
            margin=1,
        )

    def loss_function(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        positives = inputs[labels == 1]
        negatives = inputs[labels == -1]

        assert inputs.shape == positives.shape == negatives.shape
        loss = self.triplet_loss(inputs, positives, negatives)

        return loss


class BatchAllTripletLoss(nn.Module):
    def __init__(self):
        super(BatchAllTripletLoss, self).__init__()

    def loss_function(self, y: torch.Tensor, z: torch.Tensor) -> float:
        return None


class MSELoss(LossFunction):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        assert inputs.shape == targets.shape
        loss = self.mse_loss(inputs, targets)

        return loss


def get_loss_func(loss_type: str) -> LossFunction:
    if loss_type == "contrastive":
        return ContrastiveLoss()
    elif loss_type == "triplet":
        return TripletLoss()
    elif loss_type == "batch_all_triplet":
        raise NotImplementedError
        return BatchAllTripletLoss()
    elif loss_type == "mse":
        return MSELoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_type}")
