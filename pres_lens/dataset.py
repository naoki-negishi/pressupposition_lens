# Standard Library
import json
from itertools import islice

# Third Party Library
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class NLIDataset(Dataset):
    def __init__(
        self,
        raw_dataset: list[dict],
        encoder: nn.Module,
        dataset_name: str,
    ) -> None:
        self.dataset: list[dict[str, str]] = []
        self.encoder: nn.Module = encoder
        self.dataset_name: str = dataset_name

        if self.dataset_name == "imppres":
            gold_label = "gold_label"
            entailment_label = "entailment"
            contradict_label = "contradiction"
            neutral_label = "neutral"
            premise = "sentence1"
            hypothesis = "sentence2"
        elif self.dataset_name == "propres":
            gold_label = "label"
            entailment_label = "entailment"
            contradict_label = "contradiction"
            neutral_label = "neutral"
            premise = "premise"
            hypothesis = "hypothesis"
        elif self.dataset_name == "nope":
            gold_label = "label"
            entailment_label = "E"
            contradict_label = "C"
            neutral_label = "N"
            premise = "premise"
            hypothesis = "hypothesis"

        # TODO: p, h, emvironment info for detailed analysis
        dataset: list[dict[str, str]] = []
        for data in tqdm(raw_dataset, leave=False):
            if data[gold_label] == entailment_label:
                dataset.append(
                    {
                        "premise": data[premise],
                        "hypothesis": data[hypothesis],
                        "label": 1,
                    }
                )
            elif data[gold_label] in [contradict_label, neutral_label]:
                dataset.append(
                    {
                        "premise": data[premise],
                        "hypothesis": data[hypothesis],
                        "label": -1,
                    }
                )
            else:
                raise ValueError(f"Invalid gold label: {data[gold_label]}")

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        instance = self.dataset[idx]
        premise_embedding = self.encoder.encode(instance["premise"])
        hypothesis_embedding = self.encoder.encode(instance["hypothesis"])
        label = self.dataset[idx]["label"]

        return {
            "premise_embedding": premise_embedding,
            "hypothesis_embedding": hypothesis_embedding,
            "label": label,
        }

    @classmethod
    def load_and_split(
        cls,
        dataset_path: str,
        data_size_percentage: int,
        seed: int,
        sentence_encoder: nn.Module,
        dataset_name: str,
    ) -> tuple["NLIDataset", "NLIDataset"]:
        with open(dataset_path) as f:
            len_file = sum(1 for _ in f)

        with open(dataset_path) as f:
            dataset: list[dict] = []
            for jsonl in tqdm(
                islice(
                    f,
                    int(len_file * data_size_percentage / 100),
                ),
                leave=False,
            ):
                dataset.append(json.loads(jsonl))

        train_data, dev_data = train_test_split(
            dataset,
            train_size=0.7,
            shuffle=False,
            random_state=seed,
        )

        train_dataset = cls(train_data, sentence_encoder, dataset_name)
        dev_dataset = cls(dev_data, sentence_encoder, dataset_name)

        return train_dataset, dev_dataset
