"""
Loads up clusters and creates a static html site for browsing them.

The current version puts only one cluster on the page, but we'd like
there to be a way to browser all the clusters. Ideally a dropdown 
to select the cluster as well as quick navigation with arrow keys.

This version of the script generates website showing for every cluster.
A previous version only produced a website showing a single cluster.
"""

from collections import defaultdict
import pathlib
from pathlib import Path
import os
import sys
import argparse
import json

import torch


html_top = """
<!DOCTYPE html>
<html>
<head>
    <title>Clusters</title>

    <style>

    .sample {
        width: 700px;
        margin: 30px auto;
        border: 1px solid black;
        border-radius: 5px;
        padding: 10px;
    }

    .pre {
        border: 1px solid #CCC; 
        background-color: #FFF; 
        color: #000; 
        white-space: pre-wrap;
    }

    .ans {
        border: 1px solid #CCC;
        background-color: #FF9999;
        color: #000;
        white-space: pre-wrap;
    }

    .post {
        border: 1px solid #CCC;
        background-color: #FFF;
        color: #CCC;
        white-space: pre-wrap;
    }

    </style>

</head>
<body>
    <select id="cluster-selector">
        <!-- Options will be filled out by JavaScript -->
    </select>

    <div id="cluster-data">
        <!-- Data will be filled out by JavaScript -->
    </div>

    <script>

function initialize(data) {
    var selector = document.getElementById('cluster-selector');
    var display = document.getElementById('cluster-data');

    // Fill out the selector options
    data.clusters.forEach((cluster, index) => {
        var option = document.createElement('option');
        option.value = index;
        option.text = 'Cluster ' + (index + 1);
        selector.appendChild(option);
    });

    // Add event listener to change the displayed data
    selector.addEventListener('change', function() {
        var cluster = data.clusters[this.value];
        
        // Display the data as HTML
        display.innerHTML = cluster.data;
        
        // If your data is graphical, you may use a library like D3.js or Chart.js here
    });

    // Display the data of the first cluster initially
    selector.dispatchEvent(new Event('change'));

    // New code
    // Listen for arrow keys
    window.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowRight') {
            // Increment if not at last option
            if (selector.selectedIndex < selector.options.length - 1) {
                selector.selectedIndex++;
                selector.dispatchEvent(new Event('change'));
            }
        } else if (e.key === 'ArrowLeft') {
            // Decrement if not at first option
            if (selector.selectedIndex > 0) {
                selector.selectedIndex--;
                selector.dispatchEvent(new Event('change'));
            }
        }
    });
}
"""

html_bottom = """
// Define your data as a JavaScript object directly
var data = {{
    clusters: [
        {}
    ]
}};

// Call the initialize function with your data
initialize(data);

</script>

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
    parser.add_argument("--samples", type=int, default=-1, 
                        help="number of samples from the cluster to include, defaults to all of them (-1)")

    args = parser.parse_args()
    output_dir = args.output_dir
    
    clusters, _ = torch.load(args.clusters)
    # import code; code.interact(local=locals())

    clusters_html = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        # import code; code.interact(local=locals())
        cluster_html = ""
        samples = len(cluster) if args.samples == -1 else min(args.samples, len(cluster))
        for tokens, token_idx in cluster[:samples]:
            sample_html = "<div class='sample'>"
            tokens_before = min(args.before, token_idx)
            tokens_after = min(args.after, len(tokens) - token_idx - 1)
            tokens_slice = tokens[token_idx-tokens_before:token_idx + args.after + 1]
            # background_colors = [pre_answer_background_color] * tokens_before \
            #                   + [answer_token_background_color] \
            #                   + [post_answer_background_color] * tokens_after
            # text_colors = [pre_answer_text_color] * tokens_before \
            #             + [answer_token_text_color] \
            #             + [post_answer_text_color] * tokens_after
            spans_classes = ["pre"] * tokens_before \
                            + ["ans"] \
                            + ["post"] * tokens_after
            
            for token, span_class in zip(tokens_slice, spans_classes):
            # for token, background_color, text_color in zip(tokens_slice, background_colors, text_colors):
                if "\n" in token:
                    # count number of \n in token
                    num_newlines = token.count("\n")
                    for _ in range(num_newlines):
                        sample_html += f"<span class='{span_class}'> </span><br>"
                        # sample_html += f'<span style="border: 1px solid {token_border_color}; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;"> </span><br>'
                else:
                    sample_html += f"<span class='{span_class}'>{token}</span>"
                    # sample_html += f'<span style="border: 1px solid {token_border_color}; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;">{token}</span>'
            sample_html += "</div>"
            cluster_html += sample_html
        clusters_html.append(cluster_html)
    clusters_data = [{'data': cluster_html} for cluster_html in clusters_html]
    clusters_json = ',\n'.join(json.dumps(cluster) for cluster in clusters_data)
    # clusters_json = ""
    # for cluster_html in clusters_html:
    #     clusters_json += '{data: "' + cluster_html + '"},\n'
    html = html_top + html_bottom.format(clusters_json)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    clusters_filename = Path(args.clusters).parts[-1]
    clusters_filename = os.path.splitext(clusters_filename)[0]
    output_path = output_dir / f"{clusters_filename}.html"
    with open(output_path, "w") as f:
        f.write(html)
