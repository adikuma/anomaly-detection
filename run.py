import argparse
from src.config import Config
from src.train_eval import run_pipeline

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset/creditcard.csv", help="path to csv")
    p.add_argument("--latent_dim", type=int, default=8, help="latent dimensionality")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(csv_path=args.csv, latent_dim=args.latent_dim)
    results = run_pipeline(cfg)