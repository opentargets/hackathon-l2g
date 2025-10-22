# Open Targets Hackathon – Improving L2G

This repository contains code developed during the Open Targets Hackathon for the project “New ML approach for L2G problem”.
The goal is to build and evaluate a transformer-based machine learning model to address the Locus-to-Gene (L2G) problem.

L2G aims to prioritize causal genes from GWAS loci.
This project explores how modern transformer architectures can improve gene prioritization compared to original model.

## workflow detail

## Preparing input

The model uses the training data and filters out redundant features, exploiting the transformer's capacity to learn 
contextual information about neighboring genes within the same credible set.

    All "Neighbourhood" related features :
     - 'eQtlColocClppMaximumNeighbourhood',
     - 'pQtlColocClppMaximumNeighbourhood',
     - 'sQtlColocClppMaximumNeighbourhood', 
     - 'eQtlColocH4MaximumNeighbourhood',
     - 'pQtlColocH4MaximumNeighbourhood', 
     - 'sQtlColocH4MaximumNeighbourhood',
     - 'distanceSentinelFootprint', 
     - 'distanceSentinelFootprintNeighbourhood',
     - 'distanceFootprintMeanNeighbourhood',
     - 'distanceTssMeanNeighbourhood',
     - 'distanceSentinelTssNeighbourhood', 
     - 'vepMaximumNeighbourhood', 
     - 'vepMeanNeighbourhood'

     All "GeneCount" related features:
     - 'geneCount500kb'
     - 'proteinGeneCount500kb'

Feature inputs are grouped by GWAS credible sets, the positive gene within each credible set is used as the target for training. 

For training the model, it performs a 5-fold cross validation on the training set (splitting +/- 85% training, 15% training). 
Each training and testing split is done hierarchically, intending to avoid as much as possible overlapping 
positive genes between the training and testing set. 

## Trasnformer model 

The embedded gene features are passed through three stacked Transformer encoder layers. 
After passing through three Transformer layers, the output is passed through a Softmax layer that gives a probability distribution 
over all genes in the credible set. 


## Installation
Requirements

    Python 3.9+

    uv (for environment management)

### Clone the repository
    git clone https://github.com/your-username/open-targets-l2g-transformer.git
    cd open-targets-l2g-transformer

### Install uv

    pip install uv


### Dowanload input matrix

To download feature matrices:
- Train set: https://huggingface.co/opentargets/locus_to_gene_25.09/resolve/main/train.parquet
- Test set: https://huggingface.co/opentargets/locus_to_gene_25.09/resolve/main/test.parquet

Place the downloaded files in the data/ directory. 

## Usage

To get help page: 

    uv run python model.py --test

options:
| Argument          | Type    | Default  | Description                                                                                                                           |
| ----------------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--max_epochs`    | `int`   | `1000`   | Maximum number of training epochs. One epoch corresponds to a full pass through the entire training dataset.                          |
| `--patience`      | `int`   | `10`     | Number of epochs to wait for improvement in validation loss before early stopping. Helps prevent overfitting and saves training time. |
| `--batch_size`    | `int`   | `32`     | Number of samples per training batch. Larger batches can lead to faster training but may require more GPU memory.                     |
| `--learning_rate` | `float` | `3e-4`   | Learning rate for the optimizer. Controls the step size at each iteration while moving toward a minimum of the loss function.         |
| `--n_heads`       | `int`   | `2`      | Number of attention heads in each Transformer layer. More heads allow the model to focus on different representation subspaces.       |
| `--embedding_dim` | `int`   | `20`     | Size of the token embedding vectors. This controls the dimensionality of input representations to the model.                          |
| `--n_layers`      | `int`   | `3`      | Number of Transformer layers (encoder/decoder blocks). Increasing layers can improve model capacity but may increase training time.   |
| `--block_size`    | `int`   | `128`    | Maximum sequence length (context window) the model can process at once. Sequences longer than this will be truncated or split.        |
| `--device`        | `str`   | `"cuda"` | Device to run training on: `"cuda"` for GPU acceleration or `"cpu"` for CPU-only training.                                            |
| `--folds`      | `int`   | `None`      | Number of folds to use for 5-CV. Data is splitted in 5 sets, 
but this option allows you to choose how many folds you'd like to use for training.


To train the model: 

    uv run python model.py


## Deployment 

Once you have the model, you can deploy and compare it to other models using streamlit:

    uv run streamlit run app.py


## Licence 

GPL ?