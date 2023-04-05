
"""
Computes a mechanistic similarity score matrix for the given model across tokens 
from The Pile. This script does the naive thing and computes the similarity
based on the full gradients of the model (excluding embed, unembed and layernorm)
which does not scale well. Its purpose is to provide a baseline for future improved
versions to work off of. It should be sufficient to approximately reproduce the results 
from "The Quantization Model of Neural Scaling".

Note that we do not filter based on any criteria other than loss. So there is no
filtering of induction copying tokens. TODO: make sure this is okay.
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

# import scipy.linalg
import torch
import torch.nn.functional as F
# import sklearn.cluster

from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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

if __name__ == '__main__':

    # ----- define command line arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-19m")
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--cache_dir", type=str, default="~/pythia-models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pile_test_dir", type=str, default="~/the_pile")
    parser.add_argument("--losses_data", type=str, default="~/quanta-discovery/data/pythia-2.npy", 
                            help="path to a file containing the losses data")
    parser.add_argument("--loss_threshold", type=float, default=1.0, help="threshold for loss (bits) to be a candidate token")
    parser.add_argument("--skip", type=int, default=0, help="use only every `skip` tokens below the loss threshold")
    parser.add_argument("--num_tokens", type=int, default=10000, help="number of tokens to use")
    parser.add_argument("--block_len", type=int, default=250, help="number of samples to use per block when computing similarities")
    parser.add_argument("--output_dir", type=str, default="~/quanta-discovery/results")

    # ----- parse command line arguments -----
    args = parser.parse_args()
    model_name = args.model_name
    step = args.step
    device = torch.device(args.device)
    pile_test_dir = args.pile_test_dir
    losses_data = args.losses_data
    loss_threshold = args.loss_threshold
    skip = args.skip
    num_tokens = args.num_tokens
    block_len = args.block_len
    output_dir = args.output_dir

    # ----- load model and tokenizer -----
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step{step}",
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_names[0]}",
        revision=f"step{step}",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step{step}",
    ) 

    # ----- load the_pile test set -----
    dataset = load_dataset("json", data_files=os.path.join(pile_test_dir, "test.jsonl.zst"), cache_dir=pile_test_dir, split="train[:200000]") 
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step{step}",
    )

    def tokenize_sample(sample):
        tokens = tokenizer(sample["text"], return_tensors='pt', max_length=1024, truncation=True)["input_ids"]
        return {"input_ids": tokens}

    dataset = dataset.map(tokenize_sample)
    dataset = dataset.map(lambda sample: {"split_by_token": tokenizer.batch_decode(sample["input_ids"][0])})
    dataset = dataset.map(lambda sample: {"tokens_len": len(sample["input_ids"][0])})
    dataset = dataset.map(lambda sample: {"preds_len": max(sample["tokens_len"] - 1, 0)}) # fixed this on 2023-02-06 to accomodate empty documents
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

    # def print_context(idx):
    #     """
    #     given idx in range(0, 10658635), print prompt preceding the corresponding
    #     prediction, and highlight the predicted token.
    #     """
    #     sample, token_idx = get_context(idx)
    #     prompt = sample["split_by_token"][:token_idx]
    #     prompt = "".join(prompt)
    #     token = sample["split_by_token"][token_idx]
    #     print(prompt + "\033[41m" + token + "\033[0m")
        
    # ----- load losses data -----
    losses = np.load(losses_data)
    model_idx = model_names.index(model_name)
    print(losses.shape)
    losses = losses[model_idx] # TODO: check that this indexing is correct (rather than [:, model_idx])
    tokens_below_threshold = np.where(losses < loss_threshold)[0]
    token_idxs = tokens_below_threshold[::skip]
    token_idxs = token_idxs[:num_tokens]

    # ----- make the magic happen -----
    def get_flattened_gradient(model, param_subset):
        grads = []
        for name, p in model.named_parameters():
            if name in param_subset:
                grads.append(p.grad)
        return torch.cat([g.flatten() for g in grads])
    param_names = [n for n, p in model.named_parameters()]

    highsignal_names = [name for name in param_names if 
                            ('layernorm' not in name) and 
                            ('embed' not in name)]

    len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)
    S = len(token_idxs)

    blocks = [token_idxs[i:min(len(token_idxs), i+block_len)] for i in range(0, len(token_idxs), block_len)]

    C = torch.zeros((S, S), device=device)
    iouter = 0
    for iblock in tqdm(blocks):
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
        for jblock in tqdm(blocks[j_index:], leave=False):
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
    resultname = f"{model_name}_{step}_{loss_threshold}_{skip}_{num_tokens}_v1.npy"
    np.save(os.path.join(output_dir, resultname), C.cpu().numpy())


