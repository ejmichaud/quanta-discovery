
"""
Computes losses on the Pile test set, and saves them to a file.

Losses are in nats.
"""

from collections import defaultdict
import os
import argparse

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def evaluate_pile_losses(model_name, step, num_documents, cache_dir, pile_canonical, device, verbose=True):
    """Computes losses on the Pile test set, and saves them to a file.
    
    Args:
        model_name (str): name of the model to evaluate (the part following `EleutherAI/`, such as `pythia-70m`)
        step (int): step of the model to evaluate (1000 to 143000 in increments of 1000)
        num_documents (int): number of test set documents to evaluate the model on (max 200000). we only
                evaluate on at most the first 1024 tokens of each document (1023 loss values).
        cache_dir (str): directory for caching model, tokenizer, and loss values
        pile_canonical (str): path to the pre-processed pile test set
        device (str): device to use for evaluation (e.g. "cuda:0")
    """
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=os.path.join(cache_dir, model_name, f"step{step}"),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=os.path.join(cache_dir, model_name, f"step{step}"),
    )

    dataset = datasets.load_from_disk(pile_canonical)

    results = []
    for i in tqdm(range(num_documents), disable=not verbose, desc="Computing losses cache"):
        prompt = dataset[i]['text']
        if prompt:
            # this should be the same as dataset[i]['input_ids']
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device) 
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = F.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            results.append(ls.tolist())
        else:
            results.append([])
    total_length = sum(len(result) for result in results)

    results_arr = torch.zeros(total_length, dtype=torch.float32)
    j = 0
    for x in tqdm(results, leave=False, disable=not verbose, desc="Flattening results"):
        results_arr[j:j+len(x)] = torch.tensor(x, dtype=torch.float32)
        j += len(x)

    torch.save(results_arr, os.path.join(cache_dir, model_name, f"step{step}", f"{num_documents}_docs_{total_length}_tokens_losses.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="pythia-70m")
    parser.add_argument('--step', type=int, default=143000)
    parser.add_argument('--num_documents', type=int, default=20000)
    parser.add_argument('--cache_dir', type=str, 
                        default="/om/user/ericjm/quanta-discovery/cache/")
    parser.add_argument("--pile_canonical", type=str,
                        default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("-v", "--verbose", help="print progress bar", 
                        default=True, action="store_true")
    args = parser.parse_args()
    evaluate_pile_losses(**vars(args))



