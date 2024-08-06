# Standard Library
import argparse
import json
from pathlib import Path
from typing import Callable

# Third Party Library
import torch
import wandb
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

# First Party Library
from pres_lens.dataset import NLIDataset
from pres_lens.loss_function import LossFunctionForLPE
from pres_lens.models import LinearPresuppositionExtractor, LPEConfig

# from transformers import AutoModelForCausalLM, AutoTokenizer


# poetry run python pres_lens/training.py --yaml_file config/test1.yaml


def training(params_dict: dict) -> None:
    torch.manual_seed(params_dict["seed"])
    torch.cuda.manual_seed(params_dict["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model_name = params_dict["model_name"]
    if model_name == "all-MiniLM-L6-v2":
        encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        hidden_size = encoder.get_sentence_embedding_dimension()
    elif model_name == "decoder":
        raise NotImplementedError
        encoder = encoder.to(device)  # TODO: to.device でない可能性
        hidden_size = encoder.config.hidden_size
    else:
        raise ValueError(f"Invalid encoder name: {model_name}")
    encoder.zero_grad()

    # initialize LinearPresuppositionExtractor
    config = LPEConfig(
        base_model_name_or_path=params_dict["model_name"],
        d_model=hidden_size,
        bias=True,
    )
    extractor = LinearPresuppositionExtractor(config).to(device)
    loss_func = params_dict["loss_function"]
    loss_function = LossFunctionForLPE(loss_func)

    # Load datset
    dataset_path = params_dict["train_data"]
    if "IMPPRES" in dataset_path:
        dataset_name = "imppres"
    elif "PROPRES" in dataset_path:
        dataset_name = "propres"
    elif "NOPE" in dataset_path:
        dataset_name = "nope"
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    logger.info(f"Dataset name: '{dataset_name}'")
    logger.info(f"Loading dataset from '{dataset_path}'")
    with open(dataset_path) as f:
        dataset: list[dict] = []
        for line in tqdm(f):
            dataset.append(json.loads(line))
        train_dataset, dev_dataset = train_test_split(
            dataset,
            train_size=0.7,
            shuffle=True,
            random_state=params_dict["seed"],
        )

    logger.info("Creating dataloaders ...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=NLIDataset(train_dataset, encoder, dataset_name),
        batch_size=params_dict["train_batch_size"],
        shuffle=True,
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dataset=NLIDataset(dev_dataset, encoder, dataset_name),
        batch_size=params_dict["dev_batch_size"],
        shuffle=False,
    )
    logger.info(f"Train dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"Dev dataset size: {len(dev_dataloader.dataset)}")

    # training loop
    best_performance: float = 0.0  # best eval score on dev set
    num_epochs = params_dict["num_epochs"]

    logger.info(f"Start training for {num_epochs} epochs")
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
        logger.info(f"Epoch {epoch} started")

        # Training
        logger.info("Training")
        extractor.train()  # TODO: nn.Module は train() できるのか？
        logger.info("Calculating loss and accuracy on training set")
        train_loss = compute_loss(
            encoder, extractor, train_dataloader, loss_function, device
        )
        train_eval_score = evaluate(
            encoder, extractor, train_dataloader, device
        )
        logger.info(f"Train loss: {train_loss}, Train acc: {train_eval_score}")

        # Validation
        logger.info("Validation")
        extractor.eval()
        logger.info("Calculating loss and accuracy on development set")
        with torch.no_grad():
            dev_loss = compute_loss(
                encoder, extractor, dev_dataloader, loss_function, device
            )
        dev_eval_score = evaluate(encoder, extractor, dev_dataloader, device)
        logger.info(f"Dev loss: {dev_loss}, Dev acc: {dev_eval_score}")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "train_eval_score": train_eval_score,
                "dev_eval_score": dev_eval_score,
            },
            commit=True,
        )

        if dev_eval_score > best_performance:
            torch.save(
                extractor.state_dict(),
                Path(params_dict["output_dir"]) / "best_model.pth",
            )
            logger.info("Best model saved")

        logger.info(
            f"Current performance: {dev_eval_score} (Best performance: {best_performance})"
        )
        logger.info(f"Epoch {epoch} finished")


def compute_loss(
    encoder: nn.Module,
    extractor: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor, torch.device], float],
    device: torch.device,
) -> float:
    total_loss = 0.0

    for n_iter, batch in tqdm(enumerate(data_loader, 1)):
        premise_embeddings = batch["premise_embeddings"].to(device)
        hypothesis_embeddings = batch["hypothesis_embeddings"].to(device)
        assert premise_embeddings.shape == hypothesis_embeddings.shape

        y = extractor(premise_embeddings)
        loss = loss_function(y, hypothesis_embeddings)
        if extractor.training and loss != 0.0:  # TODO: != 0.0 <- need?
            loss.backward()
        else:
            loss = float(loss.cpu())
        total_loss += float(loss)

    assert n_iter == len(data_loader)
    ave_loss = total_loss / n_iter

    return ave_loss


def evaluate(encoder, extractor, data_loader, device):
    mode = "train" if extractor.training else "dev"
    total_inst_num = 0
    correct = 0

    for inst in data_loader:
        for idx, p_emb in enumerate(inst["premise_embeddings"]):
            y = extractor(p_emb.to(device))
            ys = y.expand(len(inst["hypothesis_embeddings"]), -1)


            h_embs = inst["hypothesis_embeddings"].to(device)
            cosine_scores = nn.CosineSimilarity(dim=1)(ys, h_embs)
            top = torch.argmax(cosine_scores)
            if top == idx:
                correct += 1
            total_inst_num += 1

    acc = correct / total_inst_num
    return acc

    logger.info(f"{mode} accuracy: {acc}")
    # wandb logging
    wandb.log(
        {
            f"{mode}_accuracy": acc,
        },
        commit=False,
    )


@logger.catch
def main(args: argparse.Namespace) -> None:
    params_dict = yaml.safe_load(open(args.yaml_file))

    # Setup logger
    output_dir = params_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.add(output_dir + "/training_{time}.log")
    wandb_config = {
        "yaml_file": args.yaml_file,
        "model_name": params_dict["model_name"],
        "seed": params_dict["seed"],
        "training_data": params_dict["train_data"],
        "train_batch_size": params_dict["train_batch_size"],
        "dev_batch_size": params_dict["dev_batch_size"],
        "num_train_epochs": params_dict["num_epochs"],
        "loss_function": params_dict["loss_function"],
        "learning_rate": params_dict["learning_rate"],
    }
    wandb.init(
        project=params_dict["wandb_project"],
        name=params_dict["wandb_run_name"],
        config=wandb_config,
    )

    # training
    # wandb.watch(models=encoder)  # TODO: watch what?
    training(params_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_file",
        type=str,
    )
    args = parser.parse_args()
    main(args)
