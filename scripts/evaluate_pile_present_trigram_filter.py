
"""
Computes a filter over the tokens, with the goal of eliminating tokens that
can be predicted via copying induction on their context. In the resulting array,
`True` (1) indicates that the token is the third token in a trigram that occurs
earlier in the context, and `False` (0) otherwise.

...
One could also imagine using a criteria which required the trigram to be unique,
rather than merely present, in the sense that if the first two tokens of the trigram
are the first two tokens of some other trigram in the context, then the simplest
possible induction copying scheme wouldn't uniquely determine the third token of the
trigram anymore. But this script uses the `present` criteria which filters out more tokens
than this `unique` criteria would.
"""

from collections import defaultdict
import os
import argparse

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

import datasets
from transformers import AutoTokenizer


def evaluate_pile_present_trigram_filter(model_name, step, num_documents, cache_dir, pile_canonical, verbose=True):
    """For tokens in the Pile test set determines whether they can be predicted via
    (copying) induction on their context. Saves the results as a boolean array to a file.
    
    Args:
        model_name (str): name of the model whose tokenizer to use (the part following `EleutherAI/`, such as `pythia-70m`)
        step (int): step of the model whose tokenizer to use  (1000 to 143000 in increments of 1000)
        num_documents (int): number of test set documents to evaluate the criteria on (max 200000). we only
                evaluate on at most the first 1024 tokens of each document (1023 criteria values).
        cache_dir (str): directory for caching model, tokenizer, and criteria values
        verbose (bool): whether to print progress bars
    """
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=os.path.join(cache_dir, model_name, f"step{step}"),
    )

    dataset = datasets.load_from_disk(pile_canonical)

    results = []
    for doc_i in tqdm(range(num_documents), disable=not verbose, desc="Computing ind. crit. cache"):
        prompt = dataset[doc_i]['text']
        if prompt:
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
            doc_tokens = tokens.input_ids[0].tolist()
            document_trigrams = defaultdict(int)
            criterias = []
            if len(doc_tokens) >= 2: # must be at least two tokens to have a loss
                criterias.append(False) # first predictable token cannot be predicted from induction
                for i in range(2, len(doc_tokens)):
                    trigram = tuple(doc_tokens[i-2:i+1])
                    if trigram in document_trigrams:
                        criterias.append(True)
                    else:
                        criterias.append(False)
                    document_trigrams[trigram] += 1
                results.append(criterias)
        else:
            results.append([])
    total_length = sum(len(result) for result in results)

    # save results to a flattened bool array
    results_arr = torch.zeros(total_length, dtype=torch.bool)
    j = 0
    for x in tqdm(results, leave=False, disable=not verbose, desc="Flattening results"):
        results_arr[j:j+len(x)] = torch.tensor(x, dtype=torch.bool)
        j += len(x)

    torch.save(results_arr, os.path.join(cache_dir, model_name, f"step{step}", f"{num_documents}_docs_{total_length}_tokens_present_trigram_filter.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="pythia-70m")
    parser.add_argument('--step', type=int, default=143000)
    parser.add_argument('--num_documents', type=int, default=20000)
    parser.add_argument('--cache_dir', type=str, default="/om/user/ericjm/quanta-discovery/cache/")
    parser.add_argument('--pile_canonical', type=str, default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k")
    parser.add_argument("-v", "--verbose", help="print progress bar", default=False, action="store_true")
    args = parser.parse_args()
    evaluate_pile_present_trigram_filter(**vars(args))

