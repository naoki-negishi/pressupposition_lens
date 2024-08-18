# Standard Library
import argparse
from pathlib import Path

# Third Party Library
import torch
import wandb
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm import tqdm

# First Party Library
from pres_lens.dataset import NLIDataset
from pres_lens.loss_function import LossFunction, get_loss_func
from pres_lens.models import LinearPresuppositionExtractor, LPEConfig

# poetry run python pres_lens/training.py --yaml_file config/test1.yaml


class TrainingComponents:
    params_dict: dict
    seed: int = 42
    device: torch.device

    model_name: str
    sentence_encoder: nn.Module
    hidden_size: int

    extractor: LinearPresuppositionExtractor
    loss_function: LossFunction
    learning_rate: float = 1e-3
    optimizer: torch.optim.Optimizer
    lr_gamma: float = 0.95
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR

    train_dataloader: torch.utils.data.DataLoader
    dev_dataloader: torch.utils.data.DataLoader

    num_epochs: int
    early_stopping_thres: int

    def __init__(self, params_dict: dict) -> None:
        self.params_dict = params_dict

        self.seed = self.params_dict["hyper_params"]["seed"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.set_seeds(self.seed)
        logger.info(f"Set seed: {self.seed}")

        # load sentence encoder
        self.model_name = self.params_dict["model_name"]
        logger.info(f"Loading sentence encoder: {self.model_name}")
        sentence_encoder, hidden_size = self.load_sentence_encoder(self.model_name)
        self.sentence_encoder = sentence_encoder.to(self.device)
        self.hidden_size = hidden_size
        logger.info(f"Hidden size: {self.hidden_size}")

        # initialize LinearPresuppositionExtractor
        extractor = self.init_extractor()
        self.extractor = extractor.to(self.device)
        logger.info(f"Linear Model: {str(self.extractor)}")

        # set dataloader
        dataset_path = self.params_dict["train_data"]
        data_size_percentage = self.params_dict["debug"]["data_size_percentage"]
        train_dataset, dev_dataset = self.load_and_split_dataset(
            dataset_path,
            data_size_percentage,
            self.seed,
            self.sentence_encoder,
        )
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Dev dataset size: {len(dev_dataset)}")

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.params_dict["hyper_params"]["train_batch_size"],
            shuffle=True,
        )
        self.dev_dataloader = torch.utils.data.DataLoader(
            dataset=dev_dataset,
            batch_size=self.params_dict["hyper_params"]["dev_batch_size"],
            shuffle=False,
        )

        # set
        self.num_epochs = self.params_dict["hyper_params"]["num_epochs"]
        self.early_stopping_thres = self.params_dict["hyper_params"][
            "early_stopping"
        ]

        loss_type = params_dict["hyper_params"]["loss_function"]
        logger.info(f"Loss function: {loss_type}")
        self.loss_function = get_loss_func(loss_type)

        learning_rate = float(params_dict["hyper_params"]["learning_rate"])
        logger.info(f"Initial learning rate: {learning_rate}")
        self.optimizer = torch.optim.Adam(
            self.extractor.parameters(),
            lr=learning_rate
        )
        # num_warmup_steps = (
        #     self.params["warmup_steps"]
        #     * len(self.train_dataloader)
        #     # // params.gradient_accumulation_steps
        # )
        # num_training_steps = (
        #     self.num_epochs
        #     * len(self.train_dataloader)
        #     # // params.gradient_accumulation_steps
        # )
        # self.lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )

        # self.lr_scheduler = torch.optim.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',
        #     factor=0.1,
        #     patience=5
        # )

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=10,
        #     gamma=lr_gamma
        # )

        lr_gamma = self.params_dict["hyper_params"]["lr_gamma"]
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=lr_gamma
        )

    @staticmethod
    def set_seeds(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def load_sentence_encoder(model_name: str,) -> tuple[nn.Module, int]:
        if model_name == "all-MiniLM-L6-v2":
            sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            hidden_size = sentence_encoder.get_sentence_embedding_dimension()
        elif model_name == "decoder":
            raise NotImplementedError
            # sentence_encoder = sentence_encoder.to(device)
            # hidden_size = sentence_encoder.config.hidden_size
        else:
            raise ValueError(f"Invalid sentence_encoder name: {model_name}")

        for param in sentence_encoder.parameters():
            param.requires_grad = False

        return sentence_encoder, hidden_size

    def init_extractor(self) -> LinearPresuppositionExtractor:
        lpe_config = LPEConfig(
            base_model_name_or_path=self.model_name,
            train_data_path=self.params_dict["train_data"],
            d_model=self.hidden_size,
            bias=True,
        )
        extractor = LinearPresuppositionExtractor(lpe_config)

        return extractor

    @staticmethod
    def load_and_split_dataset(
        dataset_path: str,
        data_size_percentage: int,
        seed: int,
        sentence_encoder: nn.Module
    ) -> tuple[torch.utils.data.Dataset]:
        if "IMPPRES" in dataset_path:
            dataset_name = "imppres"
        elif "PROPRES" in dataset_path:
            dataset_name = "propres"
        elif "NOPE" in dataset_path:
            dataset_name = "nope"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        logger.info(f"Loading dataset from '{dataset_path}' ...")
        logger.info(f"Using {data_size_percentage} % of the dataset")

        # **_dataset: nn.utils.data.Dataset
        train_dataset, dev_dataset = NLIDataset.load_and_split(
            dataset_path,
            data_size_percentage,
            seed,
            sentence_encoder,
            dataset_name,
        )

        return train_dataset, dev_dataset


def training(tc: TrainingComponents) -> None:
    last_loss: float = 0.0
    best_performance: float = 0.0  # best eval score on dev set
    early_stopping_count = 0

    logger.info(f"Start training for {tc.num_epochs} epochs\n")
    for epoch in range(1, tc.num_epochs + 1):
        logger.info(f"Epoch {epoch}")

        # Training
        logger.info("Training")
        tc.extractor.train()
        # logger.info("Calculating loss and accuracy on training set")
        train_loss = compute_loss(tc)
        train_eval_score = evaluate(tc)
        logger.info(f"Train loss: {last_loss:.6f} -> {train_loss:.6f}, "
                    f"Train acc: {train_eval_score:.3f}")
        last_loss = train_loss
        # logger.info(f"Train loss: {train_loss}, Train acc: {train_eval_score:.3f}")

        # Validation
        logger.info("Validation")
        tc.extractor.eval()
        # logger.info("Calculating loss and accuracy on development set")
        with torch.no_grad():
            dev_loss = compute_loss(tc)
        dev_eval_score = evaluate(tc)
        logger.info(f"Dev loss: {dev_loss:.6f}, Dev acc: {dev_eval_score:.3f} "
                    f"(Best performance: {best_performance:.3f})")

        current_lr = tc.lr_scheduler.get_last_lr()[0]
        logger.info(f"Current learning rate: {current_lr:.9f}")
        tc.lr_scheduler.step()

        if dev_eval_score > best_performance:
            best_performance = dev_eval_score
            torch.save(
                tc.extractor.state_dict(),
                Path(tc.params_dict["output_dir"]) / "best_model.pth",
            )
            logger.info("Saved best model !")

            early_stopping_count = 0
        else:
            early_stopping_count += 1
        logger.info(f"Current early stopping count: {early_stopping_count}\n")

        if tc.params_dict["wandb"]["send_to_wandb"] is True:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                    "train_eval_score": train_eval_score,
                    "dev_eval_score": dev_eval_score,
                    "best_performance": best_performance,
                    "current_lr": current_lr,
                },
                commit=True,
            )

        # if thres = 0, no early stopping
        if early_stopping_count >= tc.early_stopping_thres > 0:
            logger.info("Early stopping")
            break


def compute_loss(tc: TrainingComponents) -> float:
    total_loss = 0.0
    tc.extractor.zero_grad()
    tc.optimizer.zero_grad()
    data_loader = (
        tc.train_dataloader if tc.extractor.training else tc.dev_dataloader
    )

    for n_iter, batch in tqdm(
        enumerate(data_loader, 1), total=len(data_loader), leave=False,
        desc="Computing loss"
    ):
        premise_embeddings = batch["premise_embedding"].to(tc.device)
        hypothesis_embeddings = batch["hypothesis_embedding"].to(tc.device)
        labels = batch["label"].to(tc.device)
        assert premise_embeddings.shape == hypothesis_embeddings.shape

        outputs = tc.extractor(premise_embeddings)
        loss = tc.loss_function(outputs, hypothesis_embeddings, labels)
        if tc.extractor.training and loss != 0.0:
            loss.backward()
            tc.optimizer.step()
        total_loss += float(loss)

    assert n_iter == len(data_loader)
    ave_loss: float = total_loss / n_iter

    return ave_loss


def evaluate(tc: TrainingComponents) -> float:
    data_loader = (
        tc.train_dataloader if tc.extractor.training else tc.dev_dataloader
    )
    tc.extractor.eval()
    total_inst_num = 0
    correct = 0

    # intra-batch evaluation
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False,
                          desc="Evaluating"):
            # only deal with "label == 1: entailment" instances
            labels = batch["label"].to(tc.device)
            valid_indices = labels != -1
            if valid_indices.sum() == 0:
                continue

            p_embs = batch["premise_embedding"].to(tc.device)[valid_indices]
            h_embs = batch["hypothesis_embedding"].to(tc.device)[valid_indices]
            labels = labels[valid_indices]

            y = tc.extractor(p_embs)
            # ys: (batch_size, num_hypotheses, hidden_size)
            ys = y.unsqueeze(1).expand(-1, h_embs.size(0), -1)
            # cosine_scores: (batch_size, num_hypotheses)
            cosine_scores = nn.CosineSimilarity(dim=-1)(ys, h_embs.unsqueeze(0))
            # top_indices: (batch_size,)
            top_indices = torch.argmax(cosine_scores, dim=1)

            correct += (
                top_indices == torch.arange(len(labels), device=tc.device)
            ).sum().item()
            total_inst_num += len(labels)

    acc = correct / total_inst_num
    return acc


@logger.catch
def main(args: argparse.Namespace) -> None:
    params_dict = yaml.safe_load(open(args.yaml_file))

    # Setup logger and wandb
    output_dir = params_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.add(output_dir + "/training_{time}.log")

    wandb_config = {
        "yaml_file": args.yaml_file,
        "model_name": params_dict["model_name"],
        "training_data": params_dict["train_data"],
        "seed": params_dict["hyper_params"]["seed"],
        "num_train_epochs": params_dict["hyper_params"]["num_epochs"],
        "train_batch_size": params_dict["hyper_params"]["train_batch_size"],
        "dev_batch_size": params_dict["hyper_params"]["dev_batch_size"],
        "loss_function": params_dict["hyper_params"]["loss_function"],
        "initial_learning_rate": params_dict["hyper_params"]["learning_rate"],
    }
    if params_dict["wandb"]["send_to_wandb"] is True:
        wandb.init(
            project=params_dict["wandb"]["wandb_project"],
            name=params_dict["wandb"]["wandb_run_name"],
            config=wandb_config,
        )

    # do training
    tc = TrainingComponents(params_dict)
    training(tc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_file",
        type=str,
    )
    args = parser.parse_args()
    main(args)
