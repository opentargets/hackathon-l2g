# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from pprint import pprint
from utils import Trainer
from tqdm import tqdm
import pandas as pd
from notebooks.set_inputs import get_hierarchical_splits
import argparse
from utils import EarlyStopping

class TransformerScalarClassifier(nn.Module):

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # learned direction for scalar product
        self.scalar_vector = nn.Parameter(torch.randn(d_model))

    def forward(self, x, padding_mask=None):
        """
        x: (batch, seq_len, d_model)
        Each token represents a possible class.
        """
        # (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        # transformer encoding
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (seq_len, batch, d_model)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # scalar product per token
        logits = torch.matmul(x, self.scalar_vector)  # (batch, seq_len)

        if padding_mask is not None:
            # replace padding positions with -inf before softmax
            logits = logits.masked_fill(padding_mask, float('-inf'))

        # softmax over tokens/classes
        probs = F.softmax(logits, dim=-1)
        return logits, probs


from utils import L2GDataset, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--folds", nargs="+", default=None)
args = parser.parse_args()

block_size = args.block_size
patience = args.patience
lr = args.learning_rate
n_layers = args.n_layers
batch_size = args.batch_size
max_epochs = args.max_epochs

# %%
# seq_len = 5      # number of classes
# d_model = 20
# N_SAMPLES = 1000
# data = []
# for i in range(N_SAMPLES):
#     data.append(torch.randn(seq_len, d_model))
# 
# targets = [ random.choice(range(5)) for i in range(1000) ]

feature_matrix_train = pd.read_parquet("data/train.parquet")
#feature_matrix_test = pd.read_parquet("data/test.parquet")
#feature_matrix = pd.concat([feature_matrix_train, feature_matrix_test])

feature_matrix_train_non = feature_matrix_train.loc[:,~feature_matrix_train.columns.str.contains('Neighbourhood', case=False)]
feature_matrix_train_non = feature_matrix_train_non.loc[:,~feature_matrix_train_non.columns.str.contains('GeneCount', case=False)]

# %%
n_folds = 5
training_arrays, testing_arrays = get_hierarchical_splits(feature_matrix_train_non, n_splits=n_folds)

if args.folds is None:
    folds = list(range(n_folds))
else:
   folds = args.folds


for i in folds:
    
    training_fold = training_arrays[i]   
    feature_matrix = [ torch.tensor(training_fold[0][i][0]) for i in range(len(training_fold[0]))]
    targets = [ torch.tensor(training_fold[0][i][1]) for i in range(len(training_fold[0]))]
    labels = [ training_fold[1][i] for i in range(len(training_fold[0]))]    
    training_dataset = L2GDataset(feature_matrix, targets)


    train_loader = DataLoader(
        training_dataset, batch_size=8,
        collate_fn=lambda b: collate_fn(b, block_size)
    )

    testing_fold = testing_arrays[i]
    feature_matrix = [ torch.tensor(testing_fold[0][i][0]) for i in range(len(testing_fold[0]))]
    targets = [ torch.tensor(testing_fold[0][i][1]) for i in range(len(testing_fold[0]))]
    labels = [ testing_fold[1][i] for i in range(len(testing_fold[0]))]
    
    testing_dataset = L2GDataset(feature_matrix, targets)

    testing_loader = DataLoader(
        testing_dataset, batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, block_size)
    )
        
    model = TransformerScalarClassifier(d_model=training_dataset.n_features, n_heads=1, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)              
            
    early_stopping = EarlyStopping(patience=patience)
    trainer = Trainer(model, optimizer, train_loader, val_loader=testing_loader, device=args.device, early_stopping=early_stopping)
    trainer.train(max_epochs)

    torch.save(model.state_dict(), f"model_fold{i+1}.pt")


