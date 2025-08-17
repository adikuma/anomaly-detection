# Autoencoders

This is me trying to understand autoencoders for my second blog post :)

## Why an Autoencoder?

- It learns to reconstruct **normal** patterns.
- Unusual (fraud) patterns reconstruct poorly → high error → flagged.

## Metrics

- **Precision**: Of the flagged transactions, how many are actually fraud?
- **Recall**: Of all real frauds, how many did we catch?
- **F1**: One score that balances precision and recall.
- **AUROC** (Area Under the ROC Curve): Across all thresholds, does the model rank frauds above normals? 0.5 = random, 1.0 = perfect.
- **AUPRC** (Area Under the Precision–Recall Curve): Great for rare fraud. Baseline ≈ fraud rate.

## Are My Results Okay? I guess so :)

Example run:

- AUROC ≈ **0.964** (amazing)
- AUPRC ≈ **0.400** (very strong vs. the baseline 0.00184)
- Precision ≈ **0.292**, Recall ≈ **0.781**, F1 ≈ **0.425**

That’s pretty decent for an imbalanced fraud dataset.

## How It Works

1. Split the data: **70% train**, **10% val**, **20% test**.
2. Train the autoencoder on **normal (Class = 0)** only.
3. Compute **reconstruction error** on validation and pick a threshold.
4. Evaluate on the test set using that threshold

## Run It

```bash
pip install -r requirements.txt
python run.py --csv /path/to/creditcard.csv --latent_dim 8
```
