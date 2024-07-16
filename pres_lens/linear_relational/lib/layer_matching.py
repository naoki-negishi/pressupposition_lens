# Standard Library
import re
from collections import defaultdict
from typing import Callable, Iterable

# Third Party Library
from torch import nn

LAYER_GUESS_RE = r"^([^\d]+)\.([\d]+)(.*)$"
LayerMatcher = str | Callable[[nn.Module, int], str]


def fix_neg_layer_num(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> int:
    """Helper to handle negative layer nums. If layer_num is negative, return len(layers) + layer_num"""
    if layer_num >= 0:
        return layer_num
    matching_layers = collect_matching_layers(model, layer_matcher)
    return len(matching_layers) + layer_num


def _layer_matcher_to_callable(
    layer_matcher: LayerMatcher,
) -> Callable[[nn.Module, int], str]:
    if isinstance(layer_matcher, str):
        if "{num}" not in layer_matcher:
            raise ValueError(
                "layer_matcher must be a callable or a string containing {num}"
            )

        def matcher_callable(_model, layer_num):
            return layer_matcher.format(num=layer_num)

        # for some reason mypy doesn't like directly returning the lambda without assigning to a var first
        return matcher_callable
    return layer_matcher


def get_layer_name(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> str:
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    layer_num = fix_neg_layer_num(model, layer_matcher, layer_num)
    return matcher_callable(model, layer_num)


def guess_hidden_layer_matcher(model: nn.Module) -> str:
    """
    Guess the hidden layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_hidden_layer_matcher_from_layers(
        dict(model.named_modules()).keys()
    )


# broken into a separate function for easier testing


def _guess_hidden_layer_matcher_from_layers(layers: Iterable[str]) -> str:
    counts_by_guess: dict[str, int] = defaultdict(int)
    for layer in layers:
        if re.match(LAYER_GUESS_RE, layer):
            guess = re.sub(LAYER_GUESS_RE, r"\1.{num}\3", layer)
            counts_by_guess[guess] += 1
    if len(counts_by_guess) == 0:
        raise ValueError(
            "Could not guess hidden layer matcher, please provide a layer_matcher"
        )

    # score is higher for guesses that match more often, are and shorter in length
    guess_scores = [
        (guess, count + 1 / len(guess))
        for guess, count in counts_by_guess.items()
    ]
    return max(guess_scores, key=lambda x: x[1])[0]
