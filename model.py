# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

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


class L2GDataset(Dataset):

    def __init__(self, list_of_arrays, targets):
        self.data = list_of_arrays
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], targets[index]

    def __len__(self):
        return len(self.data)

    @property
    def n_features(self):
        return self.data[0].shape[1]
 

def collate_fn(batch, block_size):
    """
    batch: list of tuples (x, y)
        x: (seq_len_i, d_model)
        y: int (single label for the whole sample)
    """
    import torch

    max_len = block_size
    d_model = batch[0][0].shape[1]

    padded_x = []
    padding_mask = []
    labels = []

    for x, y in batch:
        pad_len = max_len - len(x)
        x_padded = torch.cat([x, torch.zeros(pad_len, d_model)], dim=0)
        mask = torch.cat([
            torch.zeros(len(x), dtype=torch.bool),
            torch.ones(pad_len, dtype=torch.bool)
        ])
        padded_x.append(x_padded)
        padding_mask.append(mask)
        labels.append(torch.tensor(y, dtype=torch.long))

    padded_x = torch.stack(padded_x)         # (batch, max_len, d_model)
    padding_mask = torch.stack(padding_mask) # (batch, max_len)
    labels = torch.stack(labels)             # (batch,)

    return padded_x, labels, padding_mask


# %%
seq_len = 5      # number of classes
d_model = 20
N_SAMPLES = 1000
data = []
for i in range(N_SAMPLES):
    data.append(torch.randn(seq_len, d_model))

targets = [ random.choice(range(5)) for i in range(1000) ]
dataset = L2GDataset(data, targets)
model = TransformerScalarClassifier(d_model=dataset.n_features, n_heads=4, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


# %%

from utils import Trainer
block_size = 128
train_loader = DataLoader(
    dataset, batch_size=8,
    collate_fn=lambda b: collate_fn(b, block_size)
)

# %%
trainer = Trainer(model, optimizer, train_loader, val_loader=None, device='cpu')
# %%

trainer.train(100)
# %%