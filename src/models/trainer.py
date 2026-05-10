"""
src/models/trainer.py
=======================
Training loop with early stopping, LR scheduling, epoch-by-epoch
checkpointing, and automatic resume if session breaks.

What this module does
---------------------
  1. Trains the model for up to max_epochs
  2. Evaluates on the validation set after every epoch
  3. Saves the best checkpoint (lowest val loss) to disk
  4. Saves a resume checkpoint after EVERY epoch
  5. Automatically resumes from last completed epoch if session breaks
  6. Stops early if val loss does not improve for patience epochs
  7. Reduces LR on plateau via ReduceLROnPlateau scheduler
  8. Supports gaussian_nll loss with uncertainty head
  9. Linear LR warmup for warmup_epochs before scheduler kicks in

Saved artefacts
---------------
  checkpoints/ALL_TICKERS_best.pt     — best val loss weights
  checkpoints/ALL_TICKERS_resume.pt   — saved every epoch for resume

config.yaml section used
------------------------
  training:
    max_epochs:     80
    patience:       15
    lr:             0.0003
    weight_decay:   0.0001
    lr_factor:      0.4
    lr_patience:    5
    grad_clip:      0.5
    batch_size:     32
    seq_len:        60
    num_workers:    2
    loss:           "gaussian_nll"
    warmup_epochs:  5
"""

import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

log = logging.getLogger(__name__)


# =============================================================================
# Loss function
# =============================================================================

def gaussian_nll_loss(mu, log_var, target):
    var = torch.exp(log_var).clamp(min=1e-6)
    return (0.5 * (log_var + (target - mu) ** 2 / var)).mean()


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


def save_resume_checkpoint(path, model, optimiser, scheduler,
                           epoch, best_val, no_improve,
                           history, cfg, feature_cols, target_col):
    """Saves full training state after every epoch for resume support."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "best_val":        best_val,
            "no_improve":      no_improve,
            "history":         history,
            "cfg":             cfg,
            "feature_cols":    feature_cols,
            "target_col":      target_col,
        },
        path,
    )


def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Checkpoint loaded <- %s  (epoch=%d  val_loss=%.6f)",
             path, ckpt["epoch"], ckpt["val_loss"])
    return ckpt


def load_resume_checkpoint(path, model, optimiser, scheduler, device="cpu"):
    """
    Loads full training state for resuming.

    Returns
    -------
    epoch, best_val, no_improve, history
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    log.info(
        "Resuming from epoch %d  (best_val=%.6f  no_improve=%d)",
        ckpt["epoch"], ckpt["best_val"], ckpt["no_improve"],
    )
    return (
        ckpt["epoch"],
        ckpt["best_val"],
        ckpt["no_improve"],
        ckpt["history"],
    )


# =============================================================================
# One epoch helpers
# =============================================================================

def _train_epoch(model, loader, criterion, use_nll,
                 optimiser, device, grad_clip):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimiser.zero_grad()

        if use_nll:
            mu, log_var = model(X_batch, return_uncertainty=True)
            loss = criterion(mu, log_var, y_batch)
        else:
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)

        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, use_nll, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if use_nll:
                mu, log_var = model(X_batch, return_uncertainty=True)
                loss = criterion(mu, log_var, y_batch)
            else:
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)

            total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


# =============================================================================
# LR warmup helper
# =============================================================================

def _warmup_lr(optimiser, epoch, warmup_epochs, base_lr):
    """Linearly ramps LR from base_lr/10 to base_lr over warmup_epochs."""
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
        for pg in optimiser.param_groups:
            pg["lr"] = max(lr, 1e-7)


# =============================================================================
# Main training function
# =============================================================================

def train(model, train_loader, val_loader,
          cfg, ckpt_path, feature_cols, target_col):
    """
    Full training loop with resume support.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    cfg          : full config dict
    ckpt_path    : path for best checkpoint  (e.g. checkpoints/ALL_TICKERS_best.pt)
    feature_cols : FEATURE_COLS
    target_col   : TARGET_COL

    Returns
    -------
    history : dict  { "train_loss", "val_loss", "lr" }
    """
    t_cfg         = cfg.get("training", {})
    max_epochs    = t_cfg.get("max_epochs",    80)
    patience      = t_cfg.get("patience",      15)
    base_lr       = t_cfg.get("lr",          3e-4)
    weight_decay  = t_cfg.get("weight_decay", 1e-4)
    lr_factor     = t_cfg.get("lr_factor",    0.4)
    lr_patience   = t_cfg.get("lr_patience",    5)
    grad_clip     = t_cfg.get("grad_clip",    0.5)
    warmup_epochs = t_cfg.get("warmup_epochs",  5)
    loss_type     = t_cfg.get("loss",         "gaussian_nll")

    use_nll   = loss_type == "gaussian_nll"
    criterion = gaussian_nll_loss if use_nll else nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s  |  loss: %s", device, loss_type)
    model.to(device)

    optimiser = Adam(model.parameters(), lr=base_lr,
                     weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min",
        factor   = lr_factor,
        patience = lr_patience,
    )

    # ── Resume checkpoint path (sibling of best checkpoint) ───────────
    resume_path = ckpt_path.replace("_best.pt", "_resume.pt")

    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_val   = float("inf")
    no_improve = 0
    start_epoch = 1

    # ── Auto-resume if resume checkpoint exists ────────────────────────
    if os.path.exists(resume_path):
        print(f"  Resume checkpoint found: {resume_path}")
        print(f"  Resuming training ...")
        start_epoch, best_val, no_improve, history = load_resume_checkpoint(
            resume_path, model, optimiser, scheduler, device
        )
        start_epoch += 1   # continue from next epoch
    else:
        print(f"  No resume checkpoint found — starting from scratch.")

    pbar = tqdm(range(start_epoch, max_epochs + 1),
                desc="Training", unit="epoch")

    for epoch in pbar:

        # ── LR warmup ─────────────────────────────────────────────────
        if epoch <= warmup_epochs:
            _warmup_lr(optimiser, epoch, warmup_epochs, base_lr)

        train_loss = _train_epoch(
            model, train_loader, criterion, use_nll,
            optimiser, device, grad_clip
        )
        val_loss   = _eval_epoch(
            model, val_loader, criterion, use_nll, device
        )
        current_lr = optimiser.param_groups[0]["lr"]

        # ── LR scheduler (only after warmup) ──────────────────────────
        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        pbar.set_postfix({
            "train": f"{train_loss:.6f}",
            "val":   f"{val_loss:.6f}",
            "lr":    f"{current_lr:.2e}",
        })

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
            tqdm.write(f"  ✓ Best model saved (epoch={epoch}  val={val_loss:.6f})")
        else:
            no_improve += 1

        # ── Save resume checkpoint every epoch ────────────────────────
        save_resume_checkpoint(
            resume_path, model, optimiser, scheduler,
            epoch, best_val, no_improve,
            history, cfg, feature_cols, target_col,
        )

        # ── Early stopping ────────────────────────────────────────────
        if no_improve >= patience:
            tqdm.write(
                f"  Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            log.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, patience,
            )
            break

    log.info(
        "Training complete.  Best val loss: %.6f  Checkpoint: %s",
        best_val, ckpt_path,
    )
    return history


# =============================================================================
# Run for all tickers  (called from pipeline.py)
# =============================================================================

def run(scaled_dict, feature_cols, target_col, cfg, ckpt_dir):
    from src.models.lstm    import build_model
    from src.models.dataset import make_loaders

    t_cfg       = cfg.get("training", {})
    batch_size  = t_cfg.get("batch_size",  32)
    seq_len     = t_cfg.get("seq_len",     60)
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
