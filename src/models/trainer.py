"""
src/models/trainer.py
=======================
Training loop with early stopping, LR scheduling, and checkpoint saving.

What this module does
---------------------
  1. Trains the model for up to max_epochs
  2. Evaluates on the validation set after every epoch
  3. Saves the best checkpoint (lowest val loss) to disk
  4. Stops early if val loss does not improve for patience epochs
  5. Reduces LR on plateau via ReduceLROnPlateau scheduler

Loss function
-------------
  MSELoss on scaled targets.
  Because the target scaler is MinMax, MSE on scaled values is
  proportional to MSE on real returns — rankings are preserved.

Saved artefacts
---------------
  checkpoints/<TICKER>_best.pt   — dict with
      {
        "model_state":      model.state_dict(),
        "epoch":            best epoch number,
        "val_loss":         best val MSE,
        "cfg":              config dict,
        "feature_cols":     FEATURE_COLS,
        "target_col":       TARGET_COL,
      }

config.yaml section used
------------------------
  training:
    max_epochs:    50
    patience:      8
    lr:            0.001
    weight_decay:  0.0001
    lr_factor:     0.5
    lr_patience:   7
    grad_clip:     1.0
    batch_size:    64
    seq_len:       30
    num_workers:   2
"""

import os
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

log = logging.getLogger(__name__)


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(path, model, epoch, val_loss, cfg,
                    feature_cols, target_col):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state":  model.state_dict(),
            "epoch":        epoch,
            "val_loss":     val_loss,
            "cfg":          cfg,
            "feature_cols": feature_cols,
            "target_col":   target_col,
        },
        path,
    )
    log.info("Checkpoint saved -> %s  (epoch=%d  val_loss=%.6f)",
             path, epoch, val_loss)


def load_checkpoint(path, model, device="cpu"):
    """
    Loads model weights from a checkpoint.

    Returns
    -------
    dict  with keys: epoch, val_loss, cfg, feature_cols, target_col
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Checkpoint loaded <- %s  (epoch=%d  val_loss=%.6f)",
             path, ckpt["epoch"], ckpt["val_loss"])
    return ckpt


# =============================================================================
# One epoch helpers
# =============================================================================

def _train_epoch(model, loader, criterion, optimiser,
                 device, grad_clip):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimiser.zero_grad()
        preds = model(X_batch)                      # (B,)
        loss  = criterion(preds, y_batch)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


# =============================================================================
# Main training function
# =============================================================================

def train(model, train_loader, val_loader,
          cfg, ckpt_path, feature_cols, target_col):
    """
    Full training loop for one model.

    Parameters
    ----------
    model        : nn.Module  (output of lstm.build_model)
    train_loader : DataLoader for the training split
    val_loader   : DataLoader for the validation split
    cfg          : full config dict (config.yaml)
    ckpt_path    : path to save the best checkpoint
    feature_cols : FEATURE_COLS  (saved in checkpoint)
    target_col   : TARGET_COL    (saved in checkpoint)

    Returns
    -------
    history : dict with lists  "train_loss", "val_loss", "lr"
    """
    t_cfg        = cfg.get("training", {})
    max_epochs   = t_cfg.get("max_epochs",   50)
    patience     = t_cfg.get("patience",      8)
    lr           = t_cfg.get("lr",          1e-3)
    weight_decay = t_cfg.get("weight_decay", 1e-4)
    lr_factor    = t_cfg.get("lr_factor",    0.5)
    lr_patience  = t_cfg.get("lr_patience",    7)
    grad_clip    = t_cfg.get("grad_clip",    1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s", device)
    model.to(device)

    criterion = nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min",
        factor   = lr_factor,
        patience = lr_patience,
    )

    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimiser, device, grad_clip
        )
        val_loss   = _eval_epoch(model, val_loader, criterion, device)
        current_lr = optimiser.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        log.info(
            "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.2e",
            epoch, max_epochs, train_loss, val_loss, current_lr,
        )

        # ── Save best checkpoint ───────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            save_checkpoint(
                ckpt_path, model, epoch, val_loss,
                cfg, feature_cols, target_col,
            )
        else:
            no_improve += 1

        # ── Early stopping ─────────────────────────────────────────────
        if no_improve >= patience:
            log.info(
                "Early stopping at epoch %d  "
                "(no improvement for %d epochs)",
                epoch, patience,
            )
            break

    log.info(
        "Training complete.  Best val loss: %.6f  "
        "Checkpoint: %s", best_val, ckpt_path,
    )
    return history


# =============================================================================
# Run for all tickers  (called from pipeline.py)
# =============================================================================

def run(scaled_dict, feature_cols, target_col,
        cfg, ckpt_dir):
    """
    Trains one model per ticker and saves checkpoints.

    Parameters
    ----------
    scaled_dict  : dict { ticker -> (train_scaled, val_scaled, test_scaled) }
                   output of scaler.run()
    feature_cols : FEATURE_COLS from features.py
    target_col   : TARGET_COL   from features.py
    cfg          : full config dict
    ckpt_dir     : directory for checkpoint files

    Returns
    -------
    history_dict : dict { ticker -> history }
    """
    from src.models.lstm    import build_model
    from src.models.dataset import make_loaders

    t_cfg       = cfg.get("training", {})
    batch_size  = t_cfg.get("batch_size",  64)
    seq_len     = t_cfg.get("seq_len",     30)
    num_workers = t_cfg.get("num_workers",  2)

    history_dict = {}

    for ticker, splits in scaled_dict.items():
        log.info("=" * 60)
        log.info("Training ticker: %s", ticker)
        log.info("=" * 60)

        single_dict = {ticker: splits}
        train_loader, val_loader, _ = make_loaders(
            single_dict, feature_cols, target_col,
            seq_len     = seq_len,
            batch_size  = batch_size,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

        model     = build_model(cfg)
        ckpt_path = os.path.join(ckpt_dir, f"{ticker}_best.pt")

        history = train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            cfg          = cfg,
            ckpt_path    = ckpt_path,
            feature_cols = feature_cols,
            target_col   = target_col,
        )

        history_dict[ticker] = history

    log.info("All tickers trained.")
    return history_dict
