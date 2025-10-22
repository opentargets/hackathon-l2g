# Open Targets Hackathon – Improving L2G

This repository contains code developed during the Open Targets Hackathon for the project “New ML approach for L2G problem”.
The goal is to build and evaluate a transformer-based machine learning model to address the Locus-to-Gene (L2G) problem.

L2G aims to prioritize causal genes from GWAS loci.
This project explores how modern transformer architectures can improve gene prioritization compared to original model.

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

To train and evaluate the model: 

python model.py


## Licence 

GPL ?