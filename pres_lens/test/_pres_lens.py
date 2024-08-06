# Standard Library
from typing import Any

# Third Party Library
import torch


def calc_lre(
    model: Any, sampled_instances: list[Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    instructions: str = ""
    few_shot_text: list[str] = []
    for inst in sampled_instances:
        premise = inst["premise"]
        hypothesis = inst["hypothesis"]

        input_text = (
            instructions + "\n".join(few_shot_text) + premise + hypothesis
        )
        premise_pos = (
            len(instructions) + len("\n".join(few_shot_text)) + len(premise)
        )
        model_outputs = model.forward(input_text)

        premise_emb = model_outputs["last_hidden"][premise_pos]
        trigger_emb = model_outputs["last_hidden"][-1]

        jacobian = torch.autograd.grad(
            trigger_emb, premise_emb, create_graph=True
        )
        lre_weight = jacobian
        lre_bias = trigger_emb - jacobian * premise_emb

    lre_weight_exp = expectation(lre_weight)
    lre_bias_exp = expectation(lre_bias)

    return lre_weight_exp, lre_bias_exp


def evaluate_lre(
    model: Any,
    datasets: list[Any],
    lre_weight: torch.Tensor,
    lre_bias: torch.Tensor,
) -> float:
    instructions: str = ""
    few_shot_text: list[str] = []
    acc = 0.0
    for inst in datasets:
        premise = inst["premise"]
        hypothesis = inst["hypothesis"]

        input_text = (
            instructions + "\n".join(few_shot_text) + premise + hypothesis
        )
        premise_pos = (
            len(instructions) + len("\n".join(few_shot_text)) + len(premise)
        )
        model_outputs = model.forward(input_text)

        # trigger_emb = model_outputs["last_hidden"][-1]
        logits = model_outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        premise_emb = model_outputs["last_hidden"][premise_pos]
        lre_logits = lre_weight * premise_emb + lre_bias
        lre_preds = torch.argmax(lre_logits, dim=-1)

    acc = (preds == lre_preds).sum() / len(inst["labels"])
    return acc


def visualize_lre(
    model: Any,
    sample_text: str,
    lre_weight: torch.Tensor,
    lre_bias: torch.Tensor,
) -> Any:

    return visualized_img


    # Calculate LRE weights and bias, and evaluate LRE
    # sampled_instances = sampling_inst(dataset, instance_num)
    # lre_weight, lre_bias = calc_lre(model, sampled_instances)
    # acc = evaluate_lre(model, dataset, lre_weight, lre_bias)
    # logger.info(f"Accuracy: {acc}")
    #
    # # Visualization
    # sample_text = params_dict["sample_text"]
    # visualized_img = visualize_lre(model, sample_text, lre_weight, lre_bias)
    #
    # # Save results
    # torch.save(lre_weight, output_dir + "/lre_weight.pth")
    # torch.save(lre_bias, output_dir + "/lre_bias.pth")
    # save_img(visualized_img, output_dir + "/visualized_img.png")
