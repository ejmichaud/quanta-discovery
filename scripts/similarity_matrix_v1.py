
"""
Computes a mechanistic similarity score matrix for the given model across tokens 
from The Pile. This script does the naive thing and computes the similarity
based on the full gradients of the model (excluding embed, unembed and layernorm)
which does not scale well. Its purpose is to provide a baseline for future improved
versions to work off of. It should be sufficient to approximately reproduce the results 
from "The Quantization Model of Neural Scaling".

Note that we do not filter based on any criteria other than loss. So there is no
filtering of induction copying tokens. TODO: make sure this is okay. Since (copying)
induction is so common, it might overwhelm the results.

Note that if filtering is needed, then we'll need to create some extra code for caching
the losses data and possibly also a filtered version of it. If a new model is specified,
then we can compute that info in this script. Otherwise we can just load it from a file.

Actually since we threshold based on loss anyway, we need to at least cache the losses
no matter what (if we want to provide compatibility with other models).

I've come up with a reasonable induction filtering scheme that is better than filtering
based on trigrams, and think that we will almost always want to filter based on this scheme.

So I think that we should have a separate script that computes whether each token hits
this criteria or not, and cache the result like we do for losses.
"""

from collections import defaultdict
import pathlib
import os
import sys
import argparse

import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from evaluate_pile_losses import evaluate_pile_losses
# from evaluate_pile_induction_criterias import evaluate_pile_induction_criterias

# import scipy.linalg
import torch
import torch.nn.functional as F
# import sklearn.cluster

import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM

if __name__ == '__main__':

    # ----- define command line arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-70m")
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--cache_dir", type=str, 
                        default="/om/user/ericjm/quanta-discovery/cache/",
                        help="directory of models, tokenizers, losses, etc.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pile_canonical", type=str,
                        default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k",
                        help="path to the canonically preprocessed Pile test set")
    parser.add_argument("--loss_threshold", type=float, default=1.0,
                        help="threshold for loss (bits) to be a candidate token")
    parser.add_argument("-f", "--filter_induction", action="store_true", 
                        help="filter induction copying tokens based on unique bigrams", 
                        default=False)
    parser.add_argument("--skip", type=int, default=1, 
                        help="use only every `skip` tokens compatible with threshold & filtering")
    parser.add_argument("--num_tokens", type=int, default=10000, 
                        help="number of tokens to use")
    parser.add_argument("--block_len", type=int, default=250, 
                        help="number of samples to use per block when computing similarities")
    parser.add_argument("--output_dir", type=str, default="~/quanta-discovery/results")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="print progress bars", default=False)

    # ----- parse command line arguments -----
    args = parser.parse_args()
    model_name = args.model_name
    step = args.step
    device = torch.device(args.device)
    cache_dir = args.cache_dir
    pile_canonical = args.pile_canonical
    loss_threshold = args.loss_threshold
    skip = args.skip
    num_tokens = args.num_tokens
    block_len = args.block_len
    output_dir = args.output_dir
    filter_induction = args.filter_induction
    verbose = args.verbose

    # ----- load model and tokenizer -----
    assert "pythia" in model_name, "must be a Pythia model"
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

    # ----- load the_pile test set -----
    dataset = datasets.load_from_disk(pile_canonical)

    def tokenize_sample(sample):
        tokens = tokenizer(sample["text"], return_tensors='pt', 
                           max_length=1024, truncation=True)["input_ids"]
        return {"input_ids": tokens}

    starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

    def loss_idx_to_dataset_idx(idx):
        """given an idx in range(0, 10658635), return
        a sample index in range(0, 20000) and pred-in-sample
        index in range(0, 1023). Note token-in-sample idx is
        exactly pred-in-sample + 1"""
        sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
        pred_in_sample_index = idx - starting_indexes[sample_index]
        return int(sample_index), int(pred_in_sample_index)

    def get_context(idx):
        """given idx in range(0, 10658635), return dataset sample
        and predicted token index within sample, in range(1, 1024)."""
        sample_index, pred_index = loss_idx_to_dataset_idx(idx)
        return dataset[sample_index], pred_index+1

    def print_context(idx):
        """
        given idx in range(0, 10658635), print prompt preceding the corresponding
        prediction, and highlight the predicted token.
        """
        sample, token_idx = get_context(idx)
        prompt = sample["split_by_token"][:token_idx]
        prompt = "".join(prompt)
        token = sample["split_by_token"][token_idx]
        print(prompt + "\033[41m" + token + "\033[0m")

 
    # ----- load losses data -----
    particular_model_cache_dir = os.path.join(cache_dir, model_name, f"step{step}")
    losses_cached = [f for f in os.listdir(particular_model_cache_dir) if f.endswith("losses.pt")]
    max_i = max(list(range(len(losses_cached))), key=lambda i: int(losses_cached[i].split("_")[0]))
    docs, tokens = int(losses_cached[max_i].split("_")[0]), int(losses_cached[max_i].split("_")[2])
    losses = torch.load(os.path.join(particular_model_cache_dir, f"{docs}_docs_{tokens}_tokens_losses.pt"))
    c = 1 / np.log(2) # nats to bits conversion

    if filter_induction:
        # we need to filter out the induction losses
        # first load the induction losses
        criterias = torch.load(os.path.join(particular_model_cache_dir, f"{docs}_docs_{tokens}_tokens_criterias.pt"))
        token_idxs = ((losses < (loss_threshold / c)) & (~criterias)).nonzero().flatten()
    else:
        token_idxs = (losses < (loss_threshold / c)).nonzero().flatten()
    token_idxs = token_idxs[::skip]
    token_idxs = token_idxs[:num_tokens].tolist()
    assert len(token_idxs) == num_tokens, "not enough tokens meeting criteria to sample from"
    
    # ----- make the magic happen -----
    def get_flattened_gradient(model, param_subset):
        grads = []
        for name, p in model.named_parameters():
            if name in param_subset:
                grads.append(p.grad)
        return torch.cat([g.flatten() for g in grads])
    param_names = [n for n, _ in model.named_parameters()]

    highsignal_names = [name for name in param_names if 
                            ('layernorm' not in name) and 
                            ('embed' not in name)]

    len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)
    S = len(token_idxs)

    blocks = [token_idxs[i:min(len(token_idxs), i+block_len)] for i in range(0, len(token_idxs), block_len)]

    C = torch.zeros((S, S), device=device)
    iouter = 0
    for iblock in tqdm(blocks, desc="outer loop", disable=not verbose):
        Gi = torch.zeros((len(iblock), len_g), device=device)
        for i, idx in enumerate(iblock):
            model.zero_grad()
            document, l = get_context(idx)
            prompt = document['text']
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            ls_l = ls[l-1]
            ls_l.backward()
            g = get_flattened_gradient(model, highsignal_names)
            # g = torch.cat([g, g.abs()])
            Gi[i] = g
        Gi = F.normalize(Gi, p=2, dim=1)
        # Gi = Gi - Gi.mean(dim=1, keepdim=True)
        j_index = blocks.index(iblock)
        jouter = sum(len(block) for block in blocks[:j_index])
        for jblock in tqdm(blocks[j_index:], leave=False, desc="inner loop", disable=not verbose):
            Gj = torch.zeros((len(jblock), len_g), device=device)
            for j, idx in enumerate(jblock):
                model.zero_grad()
                document, l = get_context(idx)
                prompt = document['text']
                tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
                logits = model(**tokens).logits
                targets = tokens.input_ids
                ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
                ls_l = ls[l-1]
                ls_l.backward()
                g = get_flattened_gradient(model, highsignal_names)
                # g = torch.cat([g, g.abs()])
                Gj[j] = g
            Gj = F.normalize(Gj, p=2, dim=1)
            # Gj = Gj - Gj.mean(dim=1, keepdim=True)
            Cij = torch.matmul(Gi, Gj.T)
            C[iouter:iouter+len(iblock), jouter:jouter+len(jblock)] = Cij
            C[jouter:jouter+len(jblock), iouter:iouter+len(iblock)] = Cij.T
            jouter += len(jblock)
        iouter += len(iblock)
 
    # ----- save the results -----
    resultname = f"{model_name}_{step}_{loss_threshold}_{skip}_{num_tokens}_v1.pt"
    torch.save((token_idxs, C.detach().cpu()), os.path.join(output_dir, resultname))


