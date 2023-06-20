"""
Loads up clusters and creates a static html site for browsing them.

The current version puts only one cluster on the page, but we'd like
there to be a way to browser all the clusters. Ideally a dropdown 
to select the cluster as well as quick navigation with arrow keys.
"""

from collections import defaultdict
import pathlib
from pathlib import Path
import os
import sys
import argparse

import torch

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Quanta Discovery</title>
    <style>

    body {{  
        
    }}

    .sample {{
        width: 700px;
        margin: 30px auto;
        border: 1px solid black;
        border-radius: 5px;
        padding: 10px;
    }}

    </style>
</head>
<body>
    <div id="container">
        {}
    </div>
</body>
</html>
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("clusters", type=str,
                         help="data file saved by clusters_from_similarity_matrix_v1.py")
    parser.add_argument("--output_dir", type=str, default="/om2/user/ericjm/quanta-discovery/results")
    parser.add_argument("--before", type=int, default=200)
    parser.add_argument("--after", type=int, default=20)
    parser.add_argument("--cluster", type=int, default=100)
    parser.add_argument("--samples", type=int, default=-1, 
                        help="number of samples from the cluster to include, defaults to all of them (-1)")

    args = parser.parse_args()
    output_dir = args.output_dir
    
    clusters = torch.load(args.clusters)
    cluster = clusters[args.cluster]

    samples = len(cluster) if args.samples == -1 else min(args.samples, len(cluster))

    # for tokens, token_idx in cluster:
    #     print("".join(tokens[:token_idx]) + f"<|{tokens[token_idx]}|>")
    #     print("-----------------------------------------------------")
    # 1 / 0
    
    token_border_color = "#CCC"
    pre_answer_background_color = "#FFF"
    answer_token_background_color = "#FF9999"
    post_answer_background_color = "#FFF"
    pre_answer_text_color = "#000"
    answer_token_text_color = "#000"
    post_answer_text_color = "#CCC"

    tokens_html = []
    for tokens, token_idx in cluster[:samples]:
        sample_html = "<div class='sample'>\n"
        tokens_before = min(args.before, token_idx)
        tokens_after = min(args.after, len(tokens) - token_idx - 1)
        tokens_slice = tokens[token_idx-tokens_before:token_idx + args.after + 1]
        background_colors = [pre_answer_background_color] * tokens_before \
                          + [answer_token_background_color] \
                          + [post_answer_background_color] * tokens_after
        text_colors = [pre_answer_text_color] * tokens_before \
                    + [answer_token_text_color] \
                    + [post_answer_text_color] * tokens_after
        
        for token, background_color, text_color in zip(tokens_slice, background_colors, text_colors):
            if "\n" in token:
                # count number of \n in token
                num_newlines = token.count("\n")
                for _ in range(num_newlines):
                    sample_html += f'<span style="border: 1px solid {token_border_color}; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;"> </span><br>'
            else:
                sample_html += f'<span style="border: 1px solid {token_border_color}; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;">{token}</span>'
        sample_html += "\n</div>"
        tokens_html.append(sample_html)
    
    html = html_template.format("\n\n\n".join(tokens_html))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    clusters_filename = Path(args.clusters).parts[-1]
    clusters_filename = os.path.splitext(clusters_filename)[0]
    output_path = output_dir / f"{clusters_filename}_cluster{args.cluster}.html"
    with open(output_path, "w") as f:
        f.write(html)
    

    # labels_sorted_by_freq = sorted(label_frequencies.keys(), key=lambda k: label_frequencies[k], reverse=True)
    # # label_permutation = [labels_sorted_by_freq.index(i) for i in labels_sorted_by_freq]
    # permutation = []
    # indices = defaultdict(list)
    # for i, cls in enumerate(clusters_labels):
    #     indices[cls].append(i)
    # for cls in labels_sorted_by_freq:
    #     permutation.extend(indices[cls])

    # matrix_filename = Path(matrix_path).parts[-1]
    # # clean the extension off
    # matrix_filename = os.path.splitext(matrix_filename)[0]
    # clusterpath = Path(output_dir) / f"{num_clusters}_{eigen_tol}_{matrix_filename}.pt"

    # clusters_data = defaultdict(list)
    # for i, label in tqdm(list(enumerate(labels_sorted_by_freq)), desc="Finding contexts", disable=not verbose):
    #     for idx_i in indices[label]:
    #         idx = token_idxs[idx_i]
    #         doc, token_idx_within_doc = get_context(idx)
    #         tokens = doc["split_by_token"]
    #         clusters_data[i].append((tokens, token_idx_within_doc))
    # torch.save(clusters_data, clusterpath)



