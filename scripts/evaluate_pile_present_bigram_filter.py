
"""
Computes a filter over the tokens, with the goal of eliminating tokens that
can be predicted via copying induction on their context. In the resulting array,
`True` (1) indicates that the token is the second token in a bigram that occurs
earlier in the context, and `False` (0) otherwise.

...
Let's try a more loose criteria for induction. Instead of the bigram
being unique, let's just require that the bigram occurs earlier in the
context. This will more aggressively filter out tokens that can be
predicted via copying induction.
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


def evaluate_pile_present_bigram_filter(model_name, step, num_documents, cache_dir, pile_canonical, verbose=True):
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
            bigram_seconds_from_first = defaultdict(list)
            criterias = []
            for i in range(1, len(doc_tokens)):
                bigram0 = doc_tokens[i-1]
                bigram1 = doc_tokens[i]
                # if bigram1 in bigram_seconds_from_first[bigram0] and len(set(bigram_seconds_from_first[bigram0])) == 1: # bigram occurs earlier and uniquely in context
                if bigram1 in bigram_seconds_from_first[bigram0]: # bigram occurs earlier in context
                    criterias.append(True) 
                else:
                    criterias.append(False)
                bigram_seconds_from_first[bigram0].append(bigram1)
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

    torch.save(results_arr, os.path.join(cache_dir, model_name, f"step{step}", f"{num_documents}_docs_{total_length}_tokens_present_bigram_filter.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="pythia-70m")
    parser.add_argument('--step', type=int, default=143000)
    parser.add_argument('--num_documents', type=int, default=20000)
    parser.add_argument('--cache_dir', type=str, default="/om/user/ericjm/quanta-discovery/cache/")
    parser.add_argument('--pile_canonical', type=str, default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k")
    parser.add_argument("-v", "--verbose", help="print progress bar", default=False, action="store_true")
    args = parser.parse_args()
    evaluate_pile_present_bigram_filter(**vars(args))




