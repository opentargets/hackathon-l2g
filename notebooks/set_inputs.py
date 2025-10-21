# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: hackathon-l2g
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import os
import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import logging

# %%
feature_matrix_train = pd.read_parquet("/home/charlotte/DRYLAB/P/OP_L2G/data/train.parquet")
#feature_matrix_test = pd.read_parquet("/home/charlotte/DRYLAB/P/OP_L2G/data/test.parquet")
#feature_matrix = pd.concat([feature_matrix_train, feature_matrix_test])
feature_matrix_train.columns

# %%
feature_matrix_train_non = feature_matrix_train.loc[:,~feature_matrix_train.columns.str.contains('Neighbourhood', case=False)] 


# %%
def hierarchical_split(
        data_df: pd.DataFrame,
        test_size: float = 0.15,
        verbose: bool = True,
        random_state: int = 777,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Implements hierarchical splitting strategy to prevent data leakage.

        Strategy:
        1. Split positives by geneId groups
        2. Further split by studyLocusId within each gene group
        3. Augment splits with corresponding negatives based on studyLocusId

        Args:
            data_df (pd.DataFrame): Input dataframe with goldStandardSet column (1=positive, 0=negative)
            test_size (float): Proportion of data for test set. Defaults to 0.15
            verbose (bool): Print splitting statistics
            random_state (int): Random seed for reproducibility. Defaults to 777

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
        """
        positives = data_df[data_df["goldStandardSet"] == 1].copy()
        negatives = data_df[data_df["goldStandardSet"] == 0].copy()

        # 1: Group positives by geneId and split genes between train/test by prioritising larger groups
        gene_groups = positives.groupby("geneId").size().reset_index(name="count")
        gene_groups = gene_groups.sort_values("count", ascending=False)

        genes_train, genes_test = train_test_split(
            gene_groups["geneId"].tolist(),
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )

        # 2: Split by studyLocusId within each gene group
        train_study_loci = set()
        test_study_loci = set()
        train_gene_positives = positives[positives["geneId"].isin(genes_train)]
        train_study_loci.update(train_gene_positives["studyLocusId"].unique())

        test_gene_positives = positives[positives["geneId"].isin(genes_test)]
        test_study_loci.update(test_gene_positives["studyLocusId"].unique())

        # If we have overlapping loci, we assign them to train set after controlling that the overlap is not too large
        overlapping_loci = train_study_loci.intersection(test_study_loci)
        if overlapping_loci:
            test_study_loci = test_study_loci - overlapping_loci
            test_gene_positives = test_gene_positives[
                ~test_gene_positives["studyLocusId"].isin(overlapping_loci)
            ]
        if len(overlapping_loci) / len(test_study_loci) > 0.1:
            logging.warning(
                "Abundant overlap between train and test sets: %d",
                len(overlapping_loci),
            )

        # Final positive splits
        train_positives = positives[positives["studyLocusId"].isin(train_study_loci)]
        test_positives = positives[positives["studyLocusId"].isin(test_study_loci)]

        if verbose:
            logging.info("Total samples: %d", len(data_df))
            logging.info("Positives: %d", len(positives))
            logging.info("Negatives: %d", len(negatives))
            logging.info("Unique genes in positives: %d", positives["geneId"].nunique())
            logging.info(
                "Unique studyLocusIds in positives: %d",
                positives["studyLocusId"].nunique(),
            )
            logging.info("\nGene-level split:")
            logging.info("Genes in train: %d", len(genes_train))
            logging.info("Genes in test: %d", len(genes_test))
            logging.info("\nStudyLocusId-level split:")
            logging.info("StudyLocusIds in train: %d", len(train_study_loci))
            logging.info("StudyLocusIds in test: %d", len(test_study_loci))
            logging.info("Positive samples in train: %d", len(train_positives))
            logging.info("Positive samples in test: %d", len(test_positives))

        # 3: Expand splits by bringing negatives to the loci
        train_negatives = negatives[negatives["studyLocusId"].isin(train_study_loci)]
        test_negatives = negatives[negatives["studyLocusId"].isin(test_study_loci)]

        # 4: Final splits
        train_df = pd.concat([train_positives, train_negatives], ignore_index=True)
        test_df = pd.concat([test_positives, test_negatives], ignore_index=True)

        train_genes = set(train_df["geneId"].unique())
        test_genes = set(test_df["geneId"].unique())
        train_loci = set(train_df["studyLocusId"].unique())
        test_loci = set(test_df["studyLocusId"].unique())
        loci_overlap = train_loci.intersection(test_loci)
        if loci_overlap:
            logging.warning(
                "Data leakage detected! Overlapping studyLocusIds between splits."
            )
        if verbose:
            gene_overlap = train_genes.intersection(test_genes)
            logging.info("\nFinal split statistics:")
            logging.info(
                "Train set: %d samples (%d positives)",
                len(train_df),
                train_df["goldStandardSet"].sum(),
            )
            logging.info(
                "Test set: %d samples (%d positives)",
                len(test_df),
                test_df["goldStandardSet"].sum(),
            )
            logging.info(
                "Gene overlap between splits (expected): %d", len(gene_overlap)
            )
            logging.info(
                "StudyLocusId overlap between splits (not expected): %d",
                len(loci_overlap),
            )

        return train_df, test_df


# %%
def split_by_group(df, 
                   group_col = "studyLocusId", 
                   y_name = 'goldStandardSet',
                   drop_group_col=True):
    """
    Split DataFrame into list of numpy arrays grouped by identifier.
    
    Args:
        df: pandas DataFrame
        group_col: column name to group by
        y_name: name of the target variable column
        drop_group_col: whether to drop the grouping column from arrays
    
    Returns:
        List of numpy arrays, one per group
    """
    arrays = []
    id_table = []
    for name, group in df.groupby(group_col):
        if drop_group_col:
            group_x = group.drop(columns=[group_col, 'geneId', y_name]).values
            group_y = group[y_name].values.argmax()  
            group_id = [name, group.iloc[group_y]["geneId"]]

        arrays.append([group_x, group_y])
        id_table.append(group_id)
    return arrays, id_table


# %%

def get_hierarchical_splits(feature_matrix_train_non: pd.DataFrame,
    n_splits: int = 1,
    random_state: int = 777,
    grouping_id: str = "studyLocusId",
    target_name: str = "goldStandardSet",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split data hierarchically and return list of array for pytorch training
    
    Args:
        df: pandas DataFrame
        random_state: random seed for reproducibility
        n_splits: number of splits to create
        grouping_id: column name to group by
        target_name: name of the target variable column
    
    Returns:
        List of numpy arrays, 2 per group, one with features, the other one with index row of best performing 
        gene within that group.
    """
    split_data_train = []
    split_data_test = []
    for fold_index in range(n_splits):
        fold_seed = random_state + fold_index
        fold_train_df, fold_val_df = hierarchical_split(
            feature_matrix_train_non,
            verbose=False,
            random_state=fold_seed
            )

        # Create numpy arrays for training
        fold_train_array = split_by_group(fold_train_df, 
                                          group_col = grouping_id,
                                          y_name = target_name)
        fold_test_array = split_by_group(fold_val_df, 
                                         group_col = grouping_id,
                                         y_name = target_name)
        split_data_train.append(fold_train_array)
        split_data_test.append(fold_test_array)

    return split_data_train, split_data_test
    


# %%
training_arrays, testing_arrays = get_hierarchical_splits(feature_matrix_train_non)


# %%

for fold in training_arrays:
    data_fold = fold[0][0]
    data_target = fold[0][1]
    data_id = fold[1]

