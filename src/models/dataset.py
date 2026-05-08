"""
src/models/dataset.py
=======================
PyTorch Dataset that converts a scaled feature DataFrame into
overlapping sliding-window sequences for LSTM training.

How it works
------------
  Given a DataFrame of T rows and F features, the dataset produces
  (T - seq_len) samples.  Sample i is:

    X[i]  = features[i : i+seq_len]     shape (seq_len, F)
    y[i]  = target[i + seq_len]          shape ()

  This is a "many-to-one" formulation:
    the LSTM sees seq_len days of history and predicts
    the NEXT day's log return.

  Example with seq_len=30:
    Row 0..29  -> predict row 30
    Row 1..30  -> predict row 31
    ...

Why sliding windows?
  LSTMs are stateless across batches. Feeding overlapping windows
  lets every training example carry a full history without manually
  managing hidden-state continuity between non-contiguous segments.

Output shapes (for a DataLoader with batch_size=B):
  X_batch : (B, seq_len, n_features)   float32 tensor
  y_batch : (B,)                       float32 tensor

Classes
-------
  PSXSequenceDataset   single-ticker sliding-window dataset
  MultiTickerDataset   pools all tickers into one flat dataset

Factories
---------
  make_loaders()       returns train / val / test DataLoaders
  make_single_loader() single-split loader (used by predict.py)
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# =============================================================================
# Single-ticker Dataset
# =============================================================================

class PSXSequenceDataset(Dataset):
    """
    Sliding-window sequence dataset for a single ticker.

    Parameters
    ----------
    df           : scaled DataFrame — output of scaler.transform_split()
                   Must contain all columns in feature_cols + target_col.
    feature_cols : ordered list of feature column names  (FEATURE_COLS)
    target_col   : name of the target column             (TARGET_COL)
    seq_len      : number of historical time steps per sample (default 30)

    Usage
    -----
    dataset = PSXSequenceDataset(train_scaled, FEATURE_COLS, TARGET_COL)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    for X_batch, y_batch in loader:
        # X_batch.shape -> (64, 30, n_features)
        # y_batch.shape -> (64,)
        ...
    """

    def __init__(self, df, feature_cols, target_col, seq_len=30):
        self.seq_len = seq_len

        # Extract numpy arrays once — __getitem__ slicing is then O(1)
        self.X = df[feature_cols].values.astype(np.float32)   # (T, F)
        self.y = df[target_col].values.astype(np.float32)     # (T,)

        self.n_samples = len(self.X) - seq_len
        if self.n_samples <= 0:
            raise ValueError(
                f"DataFrame has {len(self.X)} rows but seq_len={seq_len}. "
                f"Need at least seq_len + 1 rows."
            )

        log.info(
            "PSXSequenceDataset: %d samples  "
            "(rows=%d  seq_len=%d  features=%d)",
            self.n_samples, len(self.X), seq_len, self.X.shape[1],
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        X : torch.FloatTensor  shape (seq_len, n_features)
        y : torch.FloatTensor  scalar
        """
        x_seq = self.X[idx : idx + self.seq_len]    # (seq_len, F)
        y_val = self.y[idx + self.seq_len]           # scalar
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )


# =============================================================================
# Multi-ticker Dataset  (pools all tickers into one dataset)
# =============================================================================

class MultiTickerDataset(Dataset):
    """
    Pools sliding-window sequences from multiple tickers into a single
    dataset. During training the model sees diverse market regimes from
    all tickers simultaneously, which improves generalisation.

    Parameters
    ----------
    scaled_dict  : dict  { ticker -> (train_scaled, val_scaled, test_scaled) }
                   Output of scaler.run().
    feature_cols : list of feature column names  (FEATURE_COLS)
    target_col   : target column name            (TARGET_COL)
    split        : which split to use — "train", "val", or "test"
    seq_len      : sequence length (default 30)

    Usage
    -----
    train_set = MultiTickerDataset(scaled_dict, FEATURE_COLS, TARGET_COL,
                                   split="train", seq_len=30)
    loader    = DataLoader(train_set, batch_size=64, shuffle=True,
                           num_workers=2, pin_memory=True)
    """

    SPLIT_IDX = {"train": 0, "val": 1, "test": 2}

    def __init__(self, scaled_dict, feature_cols, target_col,
                 split="train", seq_len=30):

        if split not in self.SPLIT_IDX:
            raise ValueError(f"split must be one of {list(self.SPLIT_IDX)}")

        split_i      = self.SPLIT_IDX[split]
        self.seq_len = seq_len
        self.seqs    = []    # list of (X_array, y_array) one entry per ticker
        self.index   = []    # flat list of (ticker_idx, window_start)

        for ticker, splits in scaled_dict.items():
            df = splits[split_i]
            X  = df[feature_cols].values.astype(np.float32)
            y  = df[target_col].values.astype(np.float32)
            n  = len(X) - seq_len
            if n <= 0:
                log.warning(
                    "Skipping %s [%s]: only %d rows, need > %d",
                    ticker, split, len(X), seq_len,
                )
                continue
            t_idx = len(self.seqs)
            self.seqs.append((X, y))
            for w in range(n):
                self.index.append((t_idx, w))

        log.info(
            "MultiTickerDataset [%s]: %d tickers  %d windows  seq_len=%d",
            split, len(self.seqs), len(self.index), seq_len,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        t_idx, w     = self.index[idx]
        X_arr, y_arr = self.seqs[t_idx]
        x_seq        = X_arr[w : w + self.seq_len]
        y_val        = y_arr[w + self.seq_len]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )


# =============================================================================
# DataLoader factories
# =============================================================================

def make_loaders(scaled_dict, feature_cols, target_col,
                 seq_len=30, batch_size=64,
                 num_workers=2, pin_memory=True):
    """
    Builds train / val / test DataLoaders for all tickers pooled together.

    Parameters
    ----------
    scaled_dict  : output of scaler.run()
    feature_cols : FEATURE_COLS from features.py
    target_col   : TARGET_COL  from features.py
    seq_len      : sliding-window length  (default 30)
    batch_size   : mini-batch size        (default 64)
    num_workers  : DataLoader workers     (set 0 on Windows)
    pin_memory   : faster GPU transfer

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    common = dict(
        feature_cols = feature_cols,
        target_col   = target_col,
        seq_len      = seq_len,
    )

    train_ds = MultiTickerDataset(scaled_dict, split="train", **common)
    val_ds   = MultiTickerDataset(scaled_dict, split="val",   **common)
    test_ds  = MultiTickerDataset(scaled_dict, split="test",  **common)

    loader_cfg = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_cfg)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_cfg)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_cfg)

    log.info(
        "DataLoaders ready — train=%d  val=%d  test=%d batches (bs=%d)",
        len(train_loader), len(val_loader), len(test_loader), batch_size,
    )
    return train_loader, val_loader, test_loader


def make_single_loader(df, feature_cols, target_col,
                       seq_len=30, batch_size=64,
                       shuffle=False, num_workers=0):
    """
    Wraps a single scaled DataFrame in a DataLoader.
    Used by predict.py for per-ticker inference.

    Parameters
    ----------
    df           : scaled DataFrame (any split)
    feature_cols : FEATURE_COLS
    target_col   : TARGET_COL
    seq_len      : sliding-window length
    batch_size   : mini-batch size
    shuffle      : False for val / test / inference
    num_workers  : 0 is safe everywhere

    Returns
    -------
    DataLoader
    """
    ds = PSXSequenceDataset(df, feature_cols, target_col, seq_len=seq_len)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.feature_engineering.indicators import add_all_indicators
    from src.feature_engineering.features   import (
        build_features, FEATURE_COLS, TARGET_COL,
    )
    from src.models.scaler import scale_ticker

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    log.info("Downloading OGDC.KA ...")
    raw = yf.download("OGDC.KA", start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    df_feat = build_features(add_all_indicators(raw), sentiment_df=None)

    train_s, val_s, test_s, feat_sc, tgt_sc = scale_ticker(
        df_feat, FEATURE_COLS, TARGET_COL
    )

    # ── Single-ticker test ─────────────────────────────────────────────
    SEQ_LEN  = 30
    train_ds = PSXSequenceDataset(
        train_s, FEATURE_COLS, TARGET_COL, seq_len=SEQ_LEN
    )
    print(f"\nSingle-ticker train dataset:")
    print(f"  samples    : {len(train_ds)}")
    X0, y0 = train_ds[0]
    print(f"  X[0].shape : {X0.shape}   "
          f"(expected: ({SEQ_LEN}, {len(FEATURE_COLS)}))")
    print(f"  y[0]       : {y0.item():.6f}")

    # ── Multi-ticker test ──────────────────────────────────────────────
    scaled_dict = {"OGDC.KA": (train_s, val_s, test_s)}

    train_loader, val_loader, test_loader = make_loaders(
        scaled_dict, FEATURE_COLS, TARGET_COL,
        seq_len=SEQ_LEN, batch_size=32, num_workers=0, pin_memory=False,
    )

    print(f"\nMulti-ticker DataLoaders:")
    print(f"  train batches : {len(train_loader)}")
    print(f"  val   batches : {len(val_loader)}")
    print(f"  test  batches : {len(test_loader)}")

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  X_batch.shape : {X_batch.shape}   "
          f"(expected: (32, {SEQ_LEN}, {len(FEATURE_COLS)}))")
    print(f"  y_batch.shape : {y_batch.shape}   (expected: (32,))")
    print(f"  y range       : [{y_batch.min():.4f}, {y_batch.max():.4f}]")
    print(f"\n✓ dataset.py smoke-test passed.")
