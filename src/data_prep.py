import numpy as np
import pandas as pd

def load_dataframe(csv_path: str) -> pd.DataFrame:
    # load the csv into a dataframe
    df = pd.read_csv(csv_path)
    return df

def split_df(df: pd.DataFrame, random_state: int = 42):
    # shuffle the dataframe and split 70/10/20 into train/val/test
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(shuffled_df)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    train_df = shuffled_df.iloc[:train_end].copy()
    val_df   = shuffled_df.iloc[train_end:val_end].copy()
    test_df  = shuffled_df.iloc[val_end:].copy()
    return train_df, val_df, test_df

def log_amount(df: pd.DataFrame):
    # stabilize amount with log1p
    df.loc[:, "Amount"] = np.log1p(df["Amount"])
    return df

def fit_standardizer(train_df: pd.DataFrame, features: list[str]):
    # fit mean/std on train normals only (class == 0)
    mean = train_df[features].mean()
    std  = train_df[features].std().replace(0, 1)
    return mean, std

def apply_standardize(df: pd.DataFrame, features: list[str], mean, std):
    # apply standardization using train stats
    df.loc[:, features] = (df[features] - mean) / std
    return df
