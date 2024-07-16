# Standard Library
from dataclasses import dataclass
from typing import Generator, Iterable, Sequence, TypeVar

# Third Party Library
from tokenizers import Tokenizer
import torch
from torch import nn
from tqdm import tqdm

# Local Library
from .token_utils import (
    any_answer_matches_expected,
    get_answer_token_ids,
    predict_next_tokens_greedy,
)

T = TypeVar("T")


def get_device(model: nn.Module) -> torch.device:
    """
    Returns the device on which the model is running.
    """
    if isinstance(model.device, torch.device):
        return model.device
    return next(model.parameters()).device


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def tuplify(item: T | tuple[T, ...]) -> tuple[T, ...]:
    return item if isinstance(item, tuple) else (item,)


def shallow_flatten(items: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in items for item in sublist]


@dataclass
class AnswerMatchResult:
    prompt: str
    expected_answers: tuple[str, ...]
    potential_answers: set[str]
    answer_matches_expected: bool


def verify_answers_match_expected(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    expected_answers: Sequence[tuple[str, ...] | str],
    batch_size: int = 8,
    show_progress: bool = True,
    exact_match: bool = True,
) -> list[AnswerMatchResult]:
    if len(prompts) != len(expected_answers):
        raise ValueError(
            f"Expected {len(prompts)} prompts to match {len(expected_answers)} expected answers"
        )
    results: list[AnswerMatchResult] = []
    for batch in batchify(
        list(zip(prompts, expected_answers)), batch_size, show_progress
    ):
        batch_prompts = [prompt for prompt, _ in batch]
        batch_expected_answers = [tuplify(answers) for _, answers in batch]

        all_expected_answers = shallow_flatten(batch_expected_answers)
        tokenized_answers = [
            get_answer_token_ids(tokenizer, answer)
            for answer in all_expected_answers
        ]
        max_answer_length = max(len(tokens) for tokens in tokenized_answers)
        batch_next_tokens = predict_next_tokens_greedy(
            model,
            tokenizer,
            batch_prompts,
            num_tokens=max_answer_length,
            device=get_device(model),
        )
        for next_tokens, cur_expected_answers, cur_prompt in zip(
            batch_next_tokens, batch_expected_answers, batch_prompts
        ):
            potential_answers: set[str] = set()
            for index in range(len(next_tokens)):
                potential_answers.add(
                    tokenizer.decode(next_tokens[: index + 1])
                )
            answer_matches = any_answer_matches_expected(
                potential_answers,
                cur_expected_answers,
                exact_match=exact_match,
            )
            results.append(
                AnswerMatchResult(
                    prompt=cur_prompt,
                    expected_answers=cur_expected_answers,
                    potential_answers=potential_answers,
                    answer_matches_expected=answer_matches,
                )
            )
    return results
