# quanta-discovery
Exploring "quanta" of LLM capabilities/internals

## How to run the code

You'll first need to choose a `cache_dir` directory and a `pile-canonical` directory. The `cache_dir` directory is where we store the models, their tokenizers, loss data, etc. The `pile-canonical` directory is where we store the tokenized version of The Pile dataset.

You can populate your `pile-canonical` file by first downloading The Pile's test set at `https://the-eye.eu/public/AI/pile/test.jsonl.zst`. Then run:
```
python misc/create_pile_canonical.py --cache_dir <path/to/cache_dir> \
    --directory_containing_zst <path/to/directory_containing_zst> \
    --pile_canonical <path/to/pile-canonical>
```
This will hopefully work but I haven't tested it -- I created the script from a Jupyter notebook I had used to save The Pile formatted in this way.

What `pile-canonical` establishes is a canonical indexing into the *predictable* tokens of The Pile test set. So index 0 is the second token of the first document, which is the first token that models can generate a prediction for. We could have gotten around this by prepending a dummy token, as I've seen interpretability folks do. We could consider doing this in the future. We only include the first 1024 tokens of each document, so at most a given document has 1023 predictable tokens.

I've been using a conda environment from a previous project, so I'm not sure what the minimal requirements are, but you should be able to figure them out with some reading or trial and error. Let's soon make a `requirements.txt` file or `environment.yml` file for the repo.

Running the following scripts will approximately reproduce the results in the paper, although for some reason not exactly. I'm not sure why, but working on it. Due to randomness in the spectral clustering algorithm however, we may never be able to exactly reproduce the clusters from the paper -- it looks like I didn't set a `random_state` value for the clustering algorithm.

First we compute the model's loss values across a large number (20k) of documents in The Pile test set:
```
python scripts/evaluate_pile_losses.py --model_name pythia-70m-v0 \
    --step 143000 \
    --cache_dir <path/to/cache_dir> \
    --pile_canonical <path/to/pile-canonical> \
    --num_documents 20000 \
    --verbose
```
This will save the loss values in the same directory the model is saved at inside of `cache_dir`.

We then compute a filter which tries to identify tokens which are predictable via induction. We did this because low loss tokens are *very* typically induction tokens, and so I was worried that we would have a hard time discovering other LLM capabilities if we didn't filter out induction tokens. In the paper, I used this criteria to filter out induction tokens:
```
python scripts/evaluate_pile_present_trigram_filter.py --model_name pythia-70m-v0 \
    --step 143000 \
    --cache_dir <path/to/cache_dir> \
    --pile_canonical <path/to/pile-canonical> \
    --num_documents 20000 \
    --verbose
```
This also saves the filter in the same directory the model is saved at inside of `cache_dir`.

We now compute the pairwise cosine similarities of 10k samples:
```
python scripts/similarity_matrix_v1.py --model_name pythia-70m-v0 \
    --step 143000 \
    --cache_dir <path/to/cache_dir> \
    --pile_canonical <path/to/pile-canonical> \
    --loss_threshold 0.14426950408889636 \
    --num_documents 10000 \
    --filter <path/to/cache_dir/model_name/step/filter_name.pt> \
    --skip 50 \
    --num_tokens 10000 \
    --output_dir <path/to/output_dir> \
    --verbose
```
Note that you might need to create the output_dir before running this script. This takes me a couple hours to run on an A100. This script saves both the similarity matrix and the indices of the tokens (samples) that were used.

We then perform spectral clustering using the similarity matrix:
```
python scripts/clusters_from_similarity_matrix_v1.py </path/to/output_dir/matrix.pt> \
    --cache_dir <path/to/cache_dir> \
    --pile_canonical <path/to/pile-canonical> \
    --output_dir <path/to/output_dir> \
    --num-clusters 400 \
    --n_init 30 \
    --random_state 0 \
    --verbose
```
This saves labels for the 10k tokens as well as the documents and token index in the document for each of those tokens.

Lastly we can visualize the clusters as a website:
```
python scripts/html_from_clusters.py <path/to/output_dir/output_from_previous_script.pt> \
    --output_dir <path/to/output_dir> \
    --before 300 \
    --after 0 \
```
This saves a single html file which contains the samples for each cluster. You can navigate between clusters using left and right arrow keys or using the dropdown. Note that this file will be tens of megabytes.




