# Standard Library
import curses
import time

# Third Party Library
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# First Party Library
# from pres_lens.linear_relational.Prompt import Prompt
# from pres_lens.linear_relational.trainer import Trainer

# poetry run python pres_lens/test_lre.py --yaml_file config/test.yaml


def test():
    model_name = input("Enter the model name: ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)

    # Train LRE
    trainer = Trainer(model, tokenizer)
    prompts: list[Prompt] = []

    lre = trainer.train_lre(
        relation="located in the country of",
        subject_layer=22,
        object_layer=35,
        prompts=prompts,
    )
    print(lre)

    # Test LRE
    sample_text = input("Enter a sample text: ")
    subj_pos = int(input("Enter the position of the subject: "))
    print(f"sample_text: {sample_text}")

    inputs = tokenizer(sample_text, return_tensors="pt")

    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs["logits"][0][-1]
    print("From logits:", tokenizer.decode(logits.argmax()))

    # 8-th layer, 0-th batch
    subject_acts = outputs.hidden_states[23][0][subj_pos, :]

    object_acts_estimate = lre(subject_acts)
    ln_f = model.transformer.ln_f(object_acts_estimate)
    lm_head = model.lm_head(ln_f)
    obj_token = tokenizer.decode(lm_head.argmax())

    print(f"From LRE   : {obj_token}")
    top5 = lm_head.topk(5).indices.tolist()
    print("top 5       :", [tokenizer.decode(i) for i in top5])


def print_menu(stdscr, current_row: int, items: list[str]) -> None:
    stdscr.clear()
    stdscr.addstr(0, 0, "Select an option (press 'q' to exit):")

    for idx, item in enumerate(items, 3):
        if idx == current_row + 3:
            stdscr.addstr(idx, 0, item, curses.color_pair(1))
        else:
            stdscr.addstr(idx, 0, item)
    stdscr.refresh()

    return None


def key_input(key: str, current_row: int, max_row: int) -> tuple[int, bool]:
    is_entered = False
    if key == curses.KEY_UP and current_row > 0:
        current_row -= 1
    elif key == curses.KEY_DOWN and current_row < max_row - 1:
        current_row += 1
    elif key == ord("\n"):  # Enter
        if 0 <= current_row < max_row:
            is_entered = True
            return current_row, is_entered
    elif key == ord("q"):
        return None, False

    return current_row, is_entered


@logger.catch
def main(stdscr) -> None:
    torch.manual_seed(42)

    stdscr.nodelay(1)
    stdscr.timeout(100)

    models = [
        "gpt2",
        "facebook/opt-2.7b",
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-neox-20b",
    ]
    current_row = 0

    while True:
        print_menu(stdscr, current_row, models)

        key = stdscr.getch()
        current_row, is_entered = key_input(key, current_row, len(models))
        if current_row is None:
            curses.endwin()
            print("See you later!")
            time.sleep(1)
            return None

        if is_entered is True:
            model_name = models[current_row]
            break

    datasets = [
        "IMPPRES/all_n",
        "IMPPRES/both",
        "IMPPRES/change_of_state",
        "IMPPRES/cleft_existence",
        "IMPPRES/cleft_uniqueness",
        "IMPPRES/only",
        "IMPPRES/possessed_definite_existence",
        "IMPPRES/possessed_definite_uniqueness",
        "IMPPRES/question",
    ]
    current_row = 0

    while True:
        print_menu(stdscr, current_row, datasets)
        stdscr.addstr(1, 4, f"model_name: {model_name}")

        key = stdscr.getch()
        current_row, is_entered = key_input(key, current_row, len(datasets))
        if current_row is None:
            curses.endwin()
            print("See you later!")
            time.sleep(1)
            return None

        if is_entered is True:
            dataset_name = datasets[current_row]
            break

    curses.endwin()
    print(f"model_name: {model_name}")
    print(f"dataset_name: {dataset_name}")

    print("Testing...")
    test()


if __name__ == "__main__":
    curses.wrapper(main)
