# Standard Library
from dataclasses import dataclass
from typing import Iterable, Sequence

# Third Party Library
import torch
from tokenizers import Tokenizer
from torch import nn

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PromptAnswerData:
    answer_start_index: int
    answer: str
    answer_tokens: list[int]
    base_prompt: str
    full_prompt: str

    @property
    def num_answer_tokens(self) -> int:
        return len(self.answer_tokens)

    @property
    def output_answer_token_indices(self) -> tuple[int, ...]:
        # everything is shifted 1 earlier for output tokens
        output_start_index = self.answer_start_index - 1
        return tuple(
            range(
                output_start_index,
                output_start_index + len(self.answer_tokens),
            )
        )


def make_inputs(
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    device: torch.device = DEFAULT_DEVICE,
    add_pad_token: bool = True,
) -> dict[str, torch.Tensor]:
    ensure_tokenizer_has_pad_token(tokenizer, add_pad_token=add_pad_token)
    return tokenizer(prompts, padding=True, return_tensors="pt").to(device)


def ensure_tokenizer_has_pad_token(
    tokenizer: Tokenizer, add_pad_token: bool = True
) -> None:
    # from https://github.com/huggingface/transformers/issues/12594#issuecomment-877358955
    if not tokenizer.pad_token:
        if add_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must have a pad token")


def find_prompt_answer_data(
    tokenizer: Tokenizer, base_prompt: str, answer: str
) -> PromptAnswerData:
    """
    Find the number of tokens in the given answer, after it's appended to the prompt.
    This assumes that the answer immediately follows the prompt
    NOTE: the prompt SHOULD NOT include the answer
    """
    base_prompt_stripped = base_prompt.strip()
    answer_stripped = answer.strip()
    full_prompt = base_prompt_stripped + " " + answer_stripped
    base_prompt_tokens = tokenizer.encode(base_prompt_stripped)
    full_prompt_tokens = tokenizer.encode(full_prompt)
    return PromptAnswerData(
        answer=answer_stripped,
        answer_start_index=len(base_prompt_tokens),
        answer_tokens=full_prompt_tokens[len(base_prompt_tokens) :],
        base_prompt=base_prompt_stripped,
        full_prompt=full_prompt,
    )


def decode_tokens(
    tokenizer: Tokenizer,
    token_array: list[int] | torch.Tensor | list[torch.Tensor],
) -> list[str]:
    return [tokenizer.decode([t]) for t in token_array]


def find_all_substring_indices(
    string: str, substring: str, start: int = 0, end: int | None = None
) -> list[int]:
    """
    Find all indices of a substring in a string
    """
    indices = []
    while True:
        index = string.find(substring, start, end)
        if index == -1:
            break
        indices.append(index)
        start = index + len(substring)
    return indices


def find_token_range(
    tokenizer: Tokenizer,
    token_array: list[int] | torch.Tensor,
    substring: str,
    find_last_match: bool = True,
) -> tuple[int, int]:
    # sometimes the tokenizer messes with non-alphanumeric characters
    # so make sure the substring goes through an encoding/decoding cycle as well
    substr_toks = decode_tokens(tokenizer, tokenizer(substring)["input_ids"])
    # we want to remove the start of sentence token if the tokenizer adds it
    if tokenizer.bos_token and substr_toks[0] == tokenizer.bos_token:
        substr_toks = substr_toks[1:]
    recoded_substr = "".join(substr_toks)
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_locs = find_all_substring_indices(whole_string, recoded_substr)
    if len(char_locs) == 0:
        # sometimes adding a space in front causes different tokenization which works
        if substring[0] != " ":
            return find_token_range(tokenizer, token_array, " " + substring)
        raise ValueError(
            f"Could not find substring {recoded_substr} in {whole_string}"
        )
    token_ranges: list[tuple[int, int]] = []
    for char_loc in char_locs:
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(toks):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(recoded_substr):
                tok_end = i + 1
                break
        if tok_start is not None and tok_end is not None:
            token_ranges.append((tok_start, tok_end))
    if len(token_ranges) == 0:
        raise ValueError(
            f"Could not find substring {recoded_substr} in {toks}"
        )
    return token_ranges[-1] if find_last_match else token_ranges[0]


def find_final_word_token_index(
    tokenizer: Tokenizer, prompt: str, word: str
) -> int:
    tokens = tokenizer.encode(prompt)
    _start, end = find_token_range(tokenizer, tokens, word)
    return end - 1


def predict_from_input(
    model: nn.Module,
    inp: dict[str, torch.Tensor],
    answer_id_overrides: list[tuple[int, int]] = [],
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = predict_probs_from_input(model, inp)
    prob, pred = torch.max(probs, dim=1)
    for i, j in answer_id_overrides:
        pred[i] = j
        prob[i] = probs[i, j]
    return pred, prob


def predict_probs_from_input(
    model: nn.Module,
    inp: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = predict_logits_from_input(model, inp)
    return torch.softmax(logits, dim=-1)


def predict_logits_from_input(
    model: nn.Module,
    inp: dict[str, torch.Tensor],
) -> torch.Tensor:
    all_logits = model(**inp)["logits"]
    final_token_positions = find_final_attention_positions(
        inp["attention_mask"]
    )
    batch_indices = torch.arange(all_logits.size(0))
    return all_logits[batch_indices, final_token_positions]


def predict_next_tokens_greedy(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    num_tokens: int = 1,
    device: torch.device = DEFAULT_DEVICE,
) -> list[list[int]]:
    """
    Greedily predict the next N tokens for each prompt in the list.
    Should correctly handle right-padding.
    """
    next_prompts = [*prompts]  # copy to avoid modifying the original
    results: list[list[int]] = []
    for _i in range(num_tokens):
        # decoding and then re-encoding in a loop is wasteful, but it's the easiest way to handle
        # batches of different lengths, since model.generate() doesn't work with right-padding
        inputs = make_inputs(
            tokenizer,
            next_prompts,
            device=device,
        )
        pred_res = predict_from_input(model, inputs)
        for j, pred in enumerate(pred_res[0].detach().cpu()):
            if j >= len(results):
                results.append([])
            results[j].append(pred.item())
            next_prompts[j] += tokenizer.decode(pred)
    return results


def find_final_attention_positions(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # convoluted, and generated by ChatGPT, but seems to work
    indices = torch.arange(attention_mask.size(1)).to(attention_mask.device)
    # use broadcasting to expand indices to the shape of attention_mask
    indices = indices[None, :].expand_as(attention_mask)
    # set indices where attention_mask is 0 to -1
    indices = torch.where(attention_mask == 1, indices, -1)
    # find the max indices
    max_indices = indices.max(dim=1).values
    return max_indices


def get_answer_token_ids(
    tokenizer: Tokenizer,
    answer: str,
    ensure_space_prefix: bool = True,
    strip_start_token: bool = True,
    strip_blank_start_token: bool = True,
) -> list[int]:
    """
    Helper to find the token ids for the given answer as if it were a continuation of the prompt.
    """
    processed_answer = answer
    if ensure_space_prefix and not processed_answer.startswith(" "):
        processed_answer = " " + processed_answer
    tokens = tokenizer.encode(processed_answer)
    if strip_start_token and tokens[0] == tokenizer.bos_token_id:
        tokens = tokens[1:]
    # llama only includes an explicit space token at the start of the string if it's the first token
    if strip_blank_start_token and tokenizer.decode([tokens[0]]) == "":
        tokens = tokens[1:]
    return tokens


def any_answer_matches_expected(
    answers: Iterable[str],
    expected_answers: Iterable[str],
    exact_match: bool = True,
) -> bool:
    """
    Check if any of the given answers match any of the expected answers. Handles case and whitespace.
    """
    for answer in answers:
        if answer_matches_expected(
            answer, expected_answers, exact_match=exact_match
        ):
            return True
    return False


def answer_matches_expected(
    answer: str, expected_answers: Iterable[str], exact_match: bool = True
) -> bool:
    """
    Check if the given answer matches any of the expected answers. Handles case and whitespace.
    """
    processed_answer = process_answer(answer, exact_match)
    return processed_answer in {
        process_answer(a, exact_match) for a in expected_answers
    }


def process_answer(answer: str, exact_match: bool = True) -> str:
    """
    Process the given answer to make it easier to compare to other answers by removing case and trimming.
    """
    processed_answer = answer.strip()
    if not exact_match:
        processed_answer = processed_answer.lower()
    return processed_answer
