import numpy as np
import torch
from torch.utils.data import Dataset

class CreditCardDataset(Dataset):
    # for ae training: returns (x, x)
    def __init__(self, df, feature_cols):
        X = df[feature_cols].values.astype(np.float32)
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x  # input, target

class CreditCardEvalDataset(Dataset):
    # for evaluation: returns (x, y)
    def __init__(self, df, feature_cols):
        X = df[feature_cols].values.astype(np.float32)
        y = df["Class"].values.astype(np.int64)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
