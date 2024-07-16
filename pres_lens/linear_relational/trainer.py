# Standard Library
import random
from collections import defaultdict
from typing import Callable, Iterable, Literal, Optional, TypeVar

# Third Party Library
from linear_relational.lib.layer_matching import (
    LayerMatcher,
    guess_hidden_layer_matcher,
)
from linear_relational.Lre import Lre
from linear_relational.Prompt import Prompt
from linear_relational.PromptValidator import PromptValidator
from linear_relational.train_lre import ObjectAggregation, train_lre
from loguru import logger
from tokenizers import Tokenizer
from torch import nn

VectorAggregation = Literal["pre_mean", "post_mean"]
T = TypeVar("T")


def group_items(
    items: Iterable[T], group_fn: Callable[[T], str]
) -> dict[str, list[T]]:
    """
    Group items by the result of a function
    """
    grouped_items: dict[str, list[T]] = defaultdict(list)
    for item in items:
        group = group_fn(item)
        grouped_items[group].append(item)
    return grouped_items


def stable_shuffle(items: list[T], seed: int | float | str = 42) -> list[T]:
    """
    Shuffle a list in a stable way
    """
    generator = random.Random(seed)
    # copy items to avoid modifying original
    results = [*items]
    generator.shuffle(results)
    return results


def balance_grouped_items(
    items_by_group: dict[str, list[T]],
    max_per_group: Optional[int] = None,
    max_total: Optional[int] = None,
    seed: int | float | str = 42,
) -> list[T]:
    """
    Pick items in a round-robin fashion from each of the possible groups
    Tries to balance the amount of items that come from each group as much as possible
    `items_by_group` is a dict of group name to list of items
    """
    requests: list[T] = []
    concept_names = stable_shuffle(list(items_by_group.keys()), seed=seed)
    shuffled_reqs_by_concept = {
        concept: stable_shuffle(reqs, seed=f"{seed}{concept}")
        for concept, reqs in items_by_group.items()
    }
    prompts_per_concept: dict[str, int] = defaultdict(int)
    total_prompts = 0
    for reqs in items_by_group.values():
        num_reqs_from_concept = len(reqs)
        if max_per_group is not None and num_reqs_from_concept > max_per_group:
            num_reqs_from_concept = max_per_group
        total_prompts += num_reqs_from_concept
    if max_total is not None:
        total_prompts = min(total_prompts, max_total)

    concept_index = 0
    while len(requests) < total_prompts:
        concept_name = concept_names[concept_index]
        reqs = shuffled_reqs_by_concept[concept_name]
        concept_index = (concept_index + 1) % len(concept_names)
        if (
            max_per_group is not None
            and prompts_per_concept[concept_name] >= max_per_group
        ):
            continue
        if prompts_per_concept[concept_name] >= len(reqs):
            continue
        requests.append(reqs[prompts_per_concept[concept_name]])
        prompts_per_concept[concept_name] += 1
    return requests


class Trainer:
    """Train LREs from prompts"""

    model: nn.Module
    tokenizer: Tokenizer
    layer_matcher: LayerMatcher
    prompt_validator: PromptValidator

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        layer_matcher: Optional[LayerMatcher] = None,
        prompt_validator: Optional[PromptValidator] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_matcher = layer_matcher or guess_hidden_layer_matcher(model)
        self.prompt_validator = prompt_validator or PromptValidator(
            model, tokenizer
        )

    def train_lre(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        prompts: list[Prompt],
        max_lre_training_samples: int | None = None,
        object_aggregation: ObjectAggregation = "mean",
        validate_prompts: bool = True,
        validate_prompts_batch_size: int = 4,
        move_to_cpu: bool = False,
        verbose: bool = True,
        seed: int | str | float = 42,
    ) -> Lre:
        processed_prompts = self._process_relation_prompts(
            relation=relation,
            prompts=prompts,
            validate_prompts=validate_prompts,
            validate_prompts_batch_size=validate_prompts_batch_size,
            verbose=verbose,
        )
        prompts_by_object = group_items(
            processed_prompts, lambda p: p.object_name
        )
        lre_train_prompts = balance_grouped_items(
            items_by_group=prompts_by_object,
            max_total=max_lre_training_samples,
            seed=seed,
        )
        logger.warning(f"layer_matcher: {self.layer_matcher}")
        return train_lre(
            model=self.model,
            tokenizer=self.tokenizer,
            layer_matcher=self.layer_matcher,
            relation=relation,
            subject_layer=subject_layer,
            object_layer=object_layer,
            prompts=lre_train_prompts,
            object_aggregation=object_aggregation,
            move_to_cpu=move_to_cpu,
        )

    def _process_relation_prompts(
        self,
        relation: str,
        prompts: list[Prompt],
        validate_prompts: bool,
        validate_prompts_batch_size: int,
        verbose: bool,
    ) -> list[Prompt]:
        valid_prompts = prompts
        if validate_prompts:
            logger.info(f"validating {len(prompts)} prompts")
            valid_prompts = self.prompt_validator.filter_prompts(
                prompts, validate_prompts_batch_size, verbose
            )
        if len(valid_prompts) == 0:
            raise ValueError(f"No valid prompts found for {relation}.")
        logger.info(f"valid prompts: {len(valid_prompts)} / {len(prompts)}")
        return valid_prompts
