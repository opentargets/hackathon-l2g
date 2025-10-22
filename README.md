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

To train the model: 

    python model.py --mode train

    python model.py --mode predict


## Licence 

GPL ?