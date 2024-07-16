from typing import Any

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str) -> Any:
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    elif model_name == "gpt2-large":
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    elif model_name in ["facebook/opt-2.7b", "meta-llama/Meta-Llama-3-8B", "EleutherAI/gpt-j-6b"]:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model, tokenizer
