#!/usr/bin/env python3
"""
Learning-to-Rank Model Training Script using XGBoost

This script trains a learning-to-rank model for gene prioritization using XGBoost.
It takes training and test datasets as inputs and optionally saves the trained model.

Usage:
    python train_l2r_model.py --train_path /path/to/train.parquet --test_path /path/to/test.parquet
    python train_l2r_model.py --train_path train.parquet --test_path test.parquet --save --save_path model.skops
"""

import argparse
import pandas as pd
import numpy as np
import skops.io as sio
from sklearn.metrics import precision_score
import xgboost as xgb
import sys


# Configuration constants
GROUPCOL = "studyLocusId"
TARGET = "goldStandardSet"


def get_features(dataframe: pd.DataFrame) -> list:
    """
    Extract feature columns from the dataframe.
    
    Args:
        dataframe: Input dataframe containing features and metadata
        
    Returns:
        Sorted list of feature column names
    """
    features = [col for col in dataframe.columns if "Neighbourhood" not in col]
    features = sorted(list(set(features) - set([GROUPCOL, "geneId", TARGET])))
    return features


def process_for_l2r(data: pd.DataFrame, features: list):
    """
    Process the dataframe for Learning-to-Rank XGBoost method.
    
    Args:
        data: Input dataframe with features, labels, and group identifiers
        features: List of feature column names to use
        
    Returns:
        Tuple of (X, y, group_ids, group_sizes) as numpy arrays
    """
    # Sort data by group column to ensure proper grouping
    data = data.sort_values(by=GROUPCOL).reset_index(drop=True)
    X = data[features]
    y = data[TARGET]
    group_ids = data[GROUPCOL]
    
    # Get list of group sizes for XGBoost ranking
    unique_groups, group_counts = np.unique(group_ids, return_counts=True)
    group_sizes = group_counts.tolist()
    
    return np.array(X), np.array(y), np.array(group_ids), np.array(group_sizes)


def run_model(parameters: dict, train_df: pd.DataFrame, features: list,
              val_df: pd.DataFrame = None, save: bool = False, 
              save_path: str = None):
    """
    Train and validate the learning-to-rank model.
    
    Args:
        parameters: Dictionary of XGBoost parameters
        train_df: Training dataframe
        features: List of feature columns to use
        val_df: Optional validation dataframe
        save: Whether to save the trained model
        save_path: Path to save the model (required if save=True)
        
    Returns:
        Dictionary of evaluation metrics if val_df provided, None otherwise
    """
    print("Processing training dataframe...")
    X_train, y_train, g_train, gsize_train = process_for_l2r(train_df, features)
    
    # Create XGBoost DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(gsize_train)
    
    # Process validation data if provided
    dval = None
    if val_df is not None:
        print("Processing validation dataframe...")
        X_val, y_val, g_val, gsize_val = process_for_l2r(val_df, features)
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(gsize_val)
    
    print("Training the model...")
    # Train the model
    model = xgb.train(
        parameters,
        dtrain,
        num_boost_round=150,
        evals=[(dtrain, "train")] + ([(dval, "val")] if dval else []),
        verbose_eval=False,
    )
    
    # Save model if requested
    if save:
        if save_path is None:
            raise ValueError("Please provide a save path when save=True")
        print(f"Saving the model to {save_path}...")
        sio.dump(model, save_path)
        print("Model saved successfully!")
    
    # Evaluate on validation set if provided
    if val_df is not None:
        print("Evaluating on validation set...")
        y_pred = model.predict(dval)
        
        print("Converting predictions to top-1 per group...")
        # Select top-1 prediction per group
        y_pred_top1 = np.zeros_like(y_pred)
        for gid in np.unique(g_val):
            mask = g_val == gid
            if np.any(mask):
                top_idx = np.argmax(y_pred[mask])
                y_pred_top1[np.where(mask)[0][top_idx]] = 1
        
        # Calculate precision
        precision = precision_score(y_val, y_pred_top1)
        print("Evaluation complete!")
        
        return {
            "val_precision_top1": precision,
            "num_groups": len(np.unique(g_val)),
            "total_positives": int(y_val.sum())
        }
    
    return None


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a Learning-to-Rank model using XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with validation
  python train_l2r_model.py --train_path train.parquet --test_path test.parquet
  
  # Train and save model
  python train_l2r_model.py --train_path train.parquet --test_path test.parquet --save --save_path model.skops
  
  # Train only (no validation)
  python train_l2r_model.py --train_path train.parquet --save --save_path model.skops
        """
    )
    
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training data parquet file"
    )
    
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test/validation data parquet file (optional)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the trained model (required if --save is used)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save and args.save_path is None:
        parser.error("--save_path is required when --save is specified")
    
    # Load data
    print(f"Loading training data from {args.train_path}...")
    try:
        trainval = pd.read_parquet(args.train_path)
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
    
    test = None
    if args.test_path:
        print(f"Loading test data from {args.test_path}...")
        try:
            test = pd.read_parquet(args.test_path)
        except Exception as e:
            print(f"Error loading test data: {e}")
            sys.exit(1)
    
    # Extract features
    print("Extracting features...")
    features = get_features(trainval)
    print(f"Using {len(features)} features")
    
    # Define model parameters
    base_params = {
        "objective": "rank:map",
        "eval_metric": "map",
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    # Best params obtained with cross-validation
    best_params = {
        "eta": 0.125,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "min_child_weight": 5
    }
    
    # Merge parameters
    params = {**base_params, **best_params}
    
    # Train model
    print("\n" + "="*50)
    print("Starting model training...")
    print("="*50 + "\n")
    
    model_results = run_model(
        parameters=params,
        train_df=trainval,
        features=features,
        val_df=test,
        save=args.save,
        save_path=args.save_path
    )
    
    # Print results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    if model_results:
        print("\nValidation Results:")
        for key, value in model_results.items():
            print(f"  {key}: {value}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()