import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data_prep import (
    load_dataframe, split_df, log_amount,
    fit_standardizer, apply_standardize
)
from .datasets import CreditCardDataset, CreditCardEvalDataset
from .model import Autoencoder
from .utils import epoch_recon_loss, collect_re_and_labels, f1_from_preds

def run_pipeline(cfg: Config):
    # device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and split
    df = load_dataframe(cfg.csv_path)
    train_df, val_df, test_df = split_df(df, cfg.random_state)

    # train on normals only
    train_df = train_df[train_df["Class"] == 0].reset_index(drop=True)

    # stabilize amount
    for d in (train_df, val_df, test_df):
        log_amount(d)

    # standardize using train normals only
    mean, std = fit_standardizer(train_df, cfg.features)
    for d in (train_df, val_df, test_df):
        apply_standardize(d, cfg.features, mean, std)

    # datasets and loaders
    train_ds = CreditCardDataset(train_df, cfg.features)
    val_ds   = CreditCardEvalDataset(val_df, cfg.features)
    test_ds  = CreditCardEvalDataset(test_df, cfg.features)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size_train, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size_eval,  shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size_eval,  shuffle=False)

    # val normals for early stopping
    val_norm_df = val_df[val_df["Class"] == 0].reset_index(drop=True).copy()
    val_norm_ds = CreditCardDataset(val_norm_df, cfg.features)
    val_norm_dl = DataLoader(val_norm_ds, batch_size=max(256, cfg.batch_size_eval), shuffle=False)

    # model, loss, optimizer
    input_dim = len(cfg.features)
    model = Autoencoder(input_dim=input_dim, latent_dim=cfg.latent_dim).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training loop with early stopping
    best_val = np.inf
    best_state = None
    no_improve = 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_losses = []
        for x, _ in train_dl:
            x = x.to(device, dtype=torch.float32)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_loss = epoch_recon_loss(model, val_norm_dl, device)
        print(f"epoch {epoch:03d} | train_mse={np.mean(batch_losses):.6f} | val_norm_mse={val_loss:.6f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch} (no improvement {cfg.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    print(f"training done in {time.time()-start:.1f}s. best val_norm_mse={best_val:.6f}")

    # pick threshold on validation (mixed) by f1 sweep
    val_re, val_y = collect_re_and_labels(model, val_dl, device)
    cands = np.quantile(val_re, np.linspace(0.80, 0.999, 200))
    best = {"tau": None, "f1": -1, "prec": None, "rec": None}
    for tau in np.unique(cands):
        y_pred = (val_re >= tau).astype(int)
        prec, rec, f1 = f1_from_preds(val_y, y_pred)
        if f1 > best["f1"]:
            best = {"tau": float(tau), "f1": float(f1), "prec": float(prec), "rec": float(rec)}

    if best["tau"] is None or not np.isfinite(best["f1"]):
        # fallback: 99th percentile of val normals
        val_norm_re, _ = collect_re_and_labels(model, val_norm_dl, device)
        best["tau"] = float(np.quantile(val_norm_re, 0.99))

    print(f"chosen threshold τ={best['tau']:.6f} | val f1={best['f1']:.4f} (p={best['prec']:.4f}, r={best['rec']:.4f})")

    # test evaluation
    test_re, test_y = collect_re_and_labels(model, test_dl, device)
    test_pred = (test_re >= best["tau"]).astype(int)
    P, R, F1 = f1_from_preds(test_y, test_pred)

    results = {
        "threshold": best["tau"],
        "precision": P,
        "recall": R,
        "f1": F1,
        "fraud_rate_test": float(test_y.mean()),
        "predicted_fraud_rate": float(test_pred.mean()),
    }

    # optional auroc/auprc if sklearn is available
    try:
        from sklearn import metrics
        results["auroc"] = float(metrics.roc_auc_score(test_y, test_re))
        results["auprc"] = float(metrics.average_precision_score(test_y, test_re))
    except Exception:
        pass

    # print summary
    print("\n=== test results ===")
    print(f"threshold τ = {results['threshold']:.6f}")
    print(f"precision = {results['precision']:.4f} | recall = {results['recall']:.4f} | f1 = {results['f1']:.4f}")
    print(f"fraud rate in test: {results['fraud_rate_test']:.5f} | predicted fraud rate: {results['predicted_fraud_rate']:.5f}")
    if "auroc" in results:
        print(f"auroc = {results['auroc']:.4f} | auprc = {results['auprc']:.4f}")

    return results
