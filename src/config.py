from dataclasses import dataclass

@dataclass
class Config:
    csv_path: str = "datasets/creditcard.csv"   
    random_state: int = 42

    # training
    batch_size_train: int = 128
    batch_size_eval: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    latent_dim: int = 8  # try 4/8/16 and compare

    # features
    features = [
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
        'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
    ]