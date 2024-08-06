# Standard Library
import argparse
import json
from pathlib import Path

# Third Party Library
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

# First Party Library
from pres_lens.linear_relational.trainer import Trainer
from pres_lens.linear_relational.Prompt import Prompt

# poetry run python pres_lens/test_lre.py --yaml_file config/test.yaml


@logger.catch
def main(args: argparse.Namespace) -> None:
    params_dict = yaml.safe_load(open(args.yaml_file))

    # Setup logging
    output_dir = params_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = output_dir + "/test_{time}.log"
    logger.add(log_file)
    logger.info(f"Parameters: {params_dict}")

    # Load an autoregressive LM and tokenizer from huggingface
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = params_dict["model_name"]
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)
    print(model)

    trainer = Trainer(model, tokenizer)

    # Train LRE
    with open(params_dict["instruction_text"]) as f:
        instruction = f.read()
    with open(params_dict["train_data"]) as train_f:
        prompts = []
        for line in train_f:
            instance = json.loads(line)
            sentence1 = instance["sentence1"]
            sentence2 = instance["sentence2"]
            label = instance["gold_label"]
            if label == "entailment":
                subj = sentence1
                obj = sentence2
            else:
                continue

            prompt = Prompt(
                text=instruction,
                answer=obj,
                subject=subj,
                subject_name=subj
            )
            prompts.append(prompt)

            if len(prompts) >= params_dict["num_prompts"]:
                break

    trigger_type = params_dict["trigger_type"]
    subject_layer = params_dict["subject_layer"]
    object_layer = params_dict["object_layer"]
    lre = trainer.train_lre(
        relation=trigger_type,
        subject_layer=subject_layer,
        object_layer=object_layer,
        prompts=prompts,
    )
    print(lre)

    # evaluate LRE
    with open(params_dict["train_data"]) as train_f:
        prompts = []
        for line in train_f:
            instance = json.loads(line)
            sentence1 = instance["sentence1"]
            sentence2 = instance["sentence2"]
            label = instance["gold_label"]
            if label == "entailment":
                subj = sentence1
                obj = sentence2
            else:
                continue

            inputs = tokenizer(sample_text, return_tensors="pt")

            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs["logits"][0][-1]
            print("From logits:", tokenizer.decode(logits.argmax()))

            subject_acts = outputs.hidden_states[23][0][subj_pos, :]  # 8-th layer, 0-th batch

            object_acts_estimate = lre(subject_acts)
            ln_f = model.transformer.ln_f(object_acts_estimate)
            lm_head = model.lm_head(ln_f)
            obj_token = tokenizer.decode(lm_head.argmax())

            print(f"From LRE   : {obj_token}")
            top5 = lm_head.topk(5).indices.tolist()
            print("top 5       :", [tokenizer.decode(i) for i in top5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test LRE (training & evaluation)"
    )
    parser.add_argument(
        "--yaml_file", type=Path, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    main(args)
