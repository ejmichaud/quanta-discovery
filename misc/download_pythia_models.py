
"""
TODO: test this script
TODO: make sure it works with these model names
"""

import pathlib
import os
import sys

from tqdm.auto import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_names = [
    "pythia-19m",
    "pythia-125m",
    "pythia-350m",
    "pythia-800m",
    "pythia-1.3b",
    "pythia-2.7b",
    "pythia-6.7b",
    "pythia-13b"
]

model_names_deduped = [
    "pythia-19m-deduped",
    "pythia-125m-deduped",
    "pythia-350m-deduped",
    "pythia-800m-deduped",
    "pythia-1.3b-deduped",
    "pythia-2.7b-deduped",
    "pythia-6.7b-deduped",
    "pythia-13b-deduped"
]

if __name__ == '__main__':

    # STEPS = list(range(1000, 144000, 1000))[:-1]
    STEPS = [143000]
    MODEL_NAME = model_names[0]
    CACHE_DIR = "/om/user/ericjm/pythia-models/"
    # CACHE_DIR = os.path.join(os.path.expanduser("~"), "pythia-models")

    for step in tqdm(STEPS):
        model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{MODEL_NAME}",
        revision=f"step{step}",
        cache_dir = os.path.join(CACHE_DIR, MODEL_NAME, f"step{step}"),
        )
        del model

        tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{MODEL_NAME}",
        revision=f"step{step}",
        cache_dir = os.path.join(CACHE_DIR, MODEL_NAME, f"step{step}"),
        )
        del tokenizer



