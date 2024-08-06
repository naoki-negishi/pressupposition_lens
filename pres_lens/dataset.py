# Third Party Library
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class NLIDataset(Dataset):
    def __init__(
        self,
        dataset: list[dict],
        encoder: nn.Module,
        dataset_name: str,
    ) -> None:
        self.dataset: list[dict[str, str]] = []
        self.encoder: nn.Module = encoder
        self.dataset_name: str = dataset_name

        if self.dataset_name == "imppres":
            gold_label = "gold_label"
            entailment_label = "entailment"
            premise = "sentence1"
            hypothesis = "sentence2"
        elif self.dataset_name == "propres":
            gold_label = "label"
            entailment_label = "entailment"
            premise = "premise"
            hypothesis = "hypothesis"
        elif self.dataset_name == "nope":
            gold_label = "label"
            entailment_label = "E"
            premise = "premise"
            hypothesis = "hypothesis"

        entailment_only_dataset: list[dict[str, str]] = []
        for data in tqdm(dataset, leave=False):
            if data[gold_label] == entailment_label:
                entailment_only_dataset.append(
                    {"premise": data[premise], "hypothesis": data[hypothesis]}
                )
        self.dataset = entailment_only_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        instance = self.dataset[idx]
        premise_embeddings = self.encoder.encode(instance["premise"])
        hypothesis_embeddings = self.encoder.encode(instance["hypothesis"])

        return {
            "premise_embeddings": premise_embeddings,
            "hypothesis_embeddings": hypothesis_embeddings,
        }
