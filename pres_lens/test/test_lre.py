# Standard Library
import argparse
from pathlib import Path

# Third Party Library
import yaml
from loguru import logger

# First Party Library
from pres_lens.linear_relational.trainer import Trainer
from pres_lens.linear_relational.Prompt import Prompt
from pres_lens.models import load_model_and_tokenizer

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

    # Load a generative LM and LMHead from huggingface
    model, tokenizer = load_model_and_tokenizer(params_dict["model_name"])
    print(model)
    trainer = Trainer(model, tokenizer)

    prompts = [
        Prompt(
            "Paris is located in the country of",
            answer="France",
            subject="Paris",
        ),
        Prompt(
            "Shanghai is located in the country of",
            answer="China",
            subject="Shanghai",
        ),
        Prompt(
            "Kyoto is located in the country of",
            answer="Japan",
            subject="Kyoto",
        ),
        Prompt(
            "San Jose is located in the country of",
            answer="Costa Rica",
            subject="San Jose",
        ),
    ]

    lre = trainer.train_lre(
        relation="located in country",
        subject_layer=22,  # subject layer must be before the object layer
        object_layer=35,
        prompts=prompts,
    )
    print(lre)

    # test LRE
    sample_text = params_dict["sample_text"]
    subj_pos = params_dict["subj_pos"]
    print(f"sample_text: {sample_text}")

    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids

    outputs = model(input_ids, output_hidden_states=True)
    subject_acts = outputs.hidden_states[23][0][subj_pos, :]  # 8-th layer, 0-th batch

    object_acts_estimate = lre(subject_acts)

    print(tokenizer.decode(object_acts_estimate.argmax()))
    top5 = object_acts_estimate.topk(5).indices.tolist()
    print("top 5:", [tokenizer.decode(i) for i in top5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test LRE (training & evaluation)"
    )
    parser.add_argument(
        "--yaml_file", type=Path, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    main(args)
