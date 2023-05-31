

"""
This script loads up the similarity matrix and computes clusters from it.
It saves the clusters to a file as a list of lists of indices. It also
saves the samples (context, token pairs) corresponding to those indices
each to a file.
"""

from collections import defaultdict
import pathlib
from pathlib import Path
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
import sklearn.cluster

import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM

if __name__ == '__main__':

    # ----- define command line arguments -----
    parser = argparse.ArgumentParser()
    # add positional argument
    parser.add_argument("matrix", type=str,
                         help="data file saved by similarity_matrix_v1.py")
    # parser.add_argument("--model_name", type=str, default="pythia-70m")
    # parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--cache_dir", type=str,
                        default="/om/user/ericjm/quanta-discovery/cache/",
                        help="directory of models, tokenizers, losses, etc.")
    # parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pile_canonical", type=str,
                    default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k",
                    help="path to the canonically preprocessed Pile test set")
    # parser.add_argument("--loss_threshold", type=float, default=1.0, 
    #                     help="threshold for loss (bits) to be a candidate token")
    # parser.add_argument("-f", "--filter_induction", action="store_true", 
    #                     help="filter induction copying tokens based on unique bigrams", 
    #                     default=False)
    # parser.add_argument("--skip", type=int, default=1, 
    #                     help="use only every `skip` tokens compatible with threshold & filtering")
    # parser.add_argument("--num_tokens", type=int, default=10000, 
    #                     help="number of tokens to use")
    # parser.add_argument("--block_len", type=int, default=250, 
    #                     help="number of samples to use per block when computing similarities")
    parser.add_argument("--output_dir", type=str, default="/om2/user/ericjm/quanta-discovery/results")
    parser.add_argument("--num_clusters", type=int, default=500)
    parser.add_argument("--eigen_tol", default="auto") # TODO: add argument for clustering tolerance
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="print progress bars", default=False)

    # ----- parse command line arguments -----
    args = parser.parse_args()
    matrix_path = args.matrix
    # model_name = args.model_name
    # step = args.step
    # device = torch.device(args.device)
    cache_dir = args.cache_dir
    pile_canonical = args.pile_canonical
    # loss_threshold = args.loss_threshold
    # filter_induction = args.filter_induction
    # skip = args.skip
    # num_tokens = args.num_tokens
    # block_len = args.block_len
    output_dir = args.output_dir
    num_clusters = args.num_clusters
    eigen_tol = args.eigen_tol
    verbose = args.verbose

    # ----- load model and tokenizer -----
    # assert "pythia" in model_name, "must be a Pythia model"
    # model = GPTNeoXForCausalLM.from_pretrained(
    #     f"EleutherAI/{model_name}",
    #     revision=f"step{step}",
    #     cache_dir=os.path.join(cache_dir, model_name, f"step{step}"),
    # ).to(device)
    
    model_name = "pythia-70m"
    step = 143000
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

    token_idxs, C = torch.load(matrix_path)
    C = C.numpy()
    C = 1 - np.arccos(C) / np.pi

    if verbose:
        print("clustering...")
    clusters_labels = sklearn.cluster.SpectralClustering(n_clusters=num_clusters, 
                                                        affinity='precomputed',
                                                        eigen_tol=eigen_tol,
                                                        assign_labels='kmeans').fit_predict(C)

    label_frequencies = defaultdict(int)
    for l in clusters_labels:
        label_frequencies[l] += 1

    labels_sorted_by_freq = sorted(label_frequencies.keys(), key=lambda k: label_frequencies[k], reverse=True)
    # label_permutation = [labels_sorted_by_freq.index(i) for i in labels_sorted_by_freq]
    permutation = []
    indices = defaultdict(list)
    for i, cls in enumerate(clusters_labels):
        indices[cls].append(i)
    for cls in labels_sorted_by_freq:
        permutation.extend(indices[cls])

    matrix_filename = Path(matrix_path).parts[-1]
    clusterpath = Path(output_dir) / f"{num_clusters}_{eigen_tol}_{matrix_filename}v1.pt"

    clusters_data = defaultdict(list)
    for i, label in tqdm(list(enumerate(labels_sorted_by_freq)), desc="Finding contexts", disable=not verbose):
        for idx_i in indices[label]:
            idx = token_idxs[idx_i]
            doc, token_idx_within_doc = get_context(idx)
            tokens = doc["split_by_token"]
            clusters_data[i].append((tokens, token_idx_within_doc))
    torch.save(clusters_data, clusterpath)
