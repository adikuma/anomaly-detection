import numpy as np
import torch

@torch.no_grad()
def epoch_recon_loss(model, dataloader, device):
    # compute mean reconstruction loss over a dataloader
    model.eval()
    losses = []
    for x, _ in dataloader:
        x = x.to(device, dtype=torch.float32)
        x_hat = model(x)
        loss = ((x - x_hat) ** 2).mean(dim=1)
        losses.append(loss.detach().cpu().numpy())
    return float(np.concatenate(losses).mean())

@torch.no_grad()
def collect_re_and_labels(model, dataloader, device):
    # collect per-sample reconstruction error and labels
    model.eval()
    all_re, all_y = [], []
    for x, y in dataloader:
        x = x.to(device, dtype=torch.float32)
        x_hat = model(x)
        re = ((x - x_hat) ** 2).mean(dim=1)
        all_re.append(re.detach().cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_re), np.concatenate(all_y)

def f1_from_preds(y_true, y_pred):
    # compute precision, recall, f1
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)
