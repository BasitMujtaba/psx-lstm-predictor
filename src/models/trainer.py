"""
src/models/trainer.py
=======================
Training loop with early stopping, LR scheduling, epoch-by-epoch
checkpointing, and automatic resume if session breaks.

What this module does
---------------------
  1. Trains PSXLSTMModel (binary classification UP=1/DOWN=0)
  2. BCEWithLogitsLoss — replaces gaussian_nll (regression)
  3. Evaluates loss, accuracy, AUC, F1 on val set every epoch
  4. Saves best checkpoint (highest val AUC) to disk
  5. Saves resume checkpoint after EVERY epoch
  6. Automatically resumes from last completed epoch if session breaks
  7. Stops early if val AUC does not improve for patience epochs
  8. AdamW + CosineAnnealingLR + linear warmup
  9. Gradient clipping (grad_clip)
  10. Optional TreeEnsemble stacking after LSTM training

Saved artefacts
---------------
  checkpoints/best_model.pt    — best val AUC weights
  checkpoints/resume_model.pt  — saved every epoch for resume

config.yaml section used
------------------------
  model:
    input_size:   31
    lstm_hidden:  256
    lstm_layers:  4
    lstm_dropout: 0.3
    fc_hidden:    256
    num_heads:    8
    cross_heads:  4
    use_ensemble: true
    xgb_trees:    500
    xgb_depth:    6
    xgb_lr:       0.02
    rf_trees:     300
    rf_depth:     14

  training:
    epochs:        50
    patience:      10
    lr:            0.001
    weight_decay:  0.0001
    grad_clip:     1.0
    warmup_epochs: 3
    ckpt_dir:      /content/psx-lstm-predictor/data/checkpoints
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.models.lstm import build_model, build_ensemble

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(probs, targets, threshold=0.5):
    """
    Parameters
    ----------
    probs     : np.ndarray (N,)  probabilities in [0, 1]
    targets   : np.ndarray (N,)  true binary labels
    threshold : float

    Returns
    -------
    dict with keys: acc, auc, f1
    """
    preds = (probs >= threshold).astype(int)
    return {
        "acc": accuracy_score(targets, preds),
        "auc": roc_auc_score(targets, probs),
        "f1":  f1_score(targets, preds, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_best_checkpoint(path, model, epoch, val_auc, cfg,
                         feature_cols, target_col):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state":  model.state_dict(),
            "epoch":        epoch,
            "val_auc":      val_auc,
            "cfg":          cfg,
            "feature_cols": feature_cols,
            "target_col":   target_col,
        },
        path,
    )
    log.info("Best checkpoint saved -> %s  (epoch=%d  val_auc=%.4f)",
             path, epoch, val_auc)


def save_resume_checkpoint(path, model, optimiser, scheduler,
                           epoch, best_val_auc, no_improve,
                           history, cfg, feature_cols, target_col):
    """Saves full training state after every epoch for resume support."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "best_val_auc":    best_val_auc,
            "no_improve":      no_improve,
            "history":         history,
            "cfg":             cfg,
            "feature_cols":    feature_cols,
            "target_col":      target_col,
        },
        path,
    )


def load_resume_checkpoint(path, model, optimiser, scheduler, device="cpu"):
    """
    Loads full training state for resuming.

    Returns
    -------
    epoch, best_val_auc, no_improve, history
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    log.info(
        "Resuming from epoch %d  (best_val_auc=%.4f  no_improve=%d)",
        ckpt["epoch"], ckpt["best_val_auc"], ckpt["no_improve"],
    )
    return (
        ckpt["epoch"],
        ckpt["best_val_auc"],
        ckpt["no_improve"],
        ckpt["history"],
    )


def load_best_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Best checkpoint loaded <- %s  (epoch=%d  val_auc=%.4f)",
             path, ckpt["epoch"], ckpt["val_auc"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LR WARMUP
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_lr(optimiser, epoch, warmup_epochs, base_lr):
    """Linearly ramps LR from base_lr/10 to base_lr over warmup_epochs."""
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
        for pg in optimiser.param_groups:
            pg["lr"] = max(lr, 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ONE EPOCH — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimiser, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_probs, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()

        total_loss += loss.item() * len(yb)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(probs, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ONE EPOCH — EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits  = model(xb)
        loss    = criterion(logits, yb)

        total_loss += loss.item() * len(yb)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(probs, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, probs, targets


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader,
          cfg, ckpt_dir, feature_cols, target_col):
    """
    Full training loop with resume support.

    Parameters
    ----------
    model        : PSXLSTMModel
    train_loader : DataLoader
    val_loader   : DataLoader
    cfg          : dict  (loaded config.yaml)
    ckpt_dir     : str   directory to save checkpoints
    feature_cols : FEATURE_COLS
    target_col   : TARGET_COL

    Returns
    -------
    history : dict { train_loss, train_acc, train_auc, train_f1,
                     val_loss,   val_acc,   val_auc,   val_f1, lr }
    """
    t_cfg         = cfg.get("training", {})
    epochs        = t_cfg.get("epochs",        50)
    patience      = t_cfg.get("patience",      10)
    base_lr       = t_cfg.get("lr",          1e-3)
    weight_decay  = t_cfg.get("weight_decay", 1e-4)
    grad_clip     = t_cfg.get("grad_clip",    1.0)
    warmup_epochs = t_cfg.get("warmup_epochs",  3)

    best_ckpt   = os.path.join(ckpt_dir, "best_model.pt")
    resume_ckpt = os.path.join(ckpt_dir, "resume_model.pt")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimiser = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "train_acc": [], "train_auc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [], "val_f1":   [],
        "lr":         [],
    }
    best_val_auc = 0.0
    no_improve   = 0
    start_epoch  = 1

    # ── Auto-resume ───────────────────────────────────────────────────
    if os.path.exists(resume_ckpt):
        print("  Resume checkpoint found: " + resume_ckpt)
        start_epoch, best_val_auc, no_improve, history = load_resume_checkpoint(
            resume_ckpt, model, optimiser, scheduler, device
        )
        start_epoch += 1
    else:
        print("  No resume checkpoint found — starting from scratch.")

    # ── Header ────────────────────────────────────────────────────────
    header = ("  " +
              "Ep".rjust(4)  + "  " +
              "TrLoss".rjust(7) + "  " +
              "TrAcc".rjust(6)  + "  " +
              "TrAUC".rjust(6)  + "  " +
              "TrF1".rjust(6)   + "  " +
              "VlLoss".rjust(7) + "  " +
              "VlAcc".rjust(6)  + "  " +
              "VlAUC".rjust(6)  + "  " +
              "VlF1".rjust(6)   + "  " +
              "LR".rjust(9))
    print("\n" + header)
    print("  " + "-" * 90)

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:

        # ── LR warmup ─────────────────────────────────────────────────
        if epoch <= warmup_epochs:
            _warmup_lr(optimiser, epoch, warmup_epochs, base_lr)

        train_m          = _train_epoch(model, train_loader, criterion,
                                        optimiser, device, grad_clip)
        val_m, _, _      = _eval_epoch(model, val_loader, criterion, device)
        current_lr       = optimiser.param_groups[0]["lr"]

        # ── LR scheduler (only after warmup) ──────────────────────────
        if epoch > warmup_epochs:
            scheduler.step()

        # ── Record history ────────────────────────────────────────────
        for k in ["loss", "acc", "auc", "f1"]:
            history["train_" + k].append(train_m[k])
            history["val_"   + k].append(val_m[k])
        history["lr"].append(current_lr)

        # ── Best checkpoint ───────────────────────────────────────────
        improved = val_m["auc"] > best_val_auc
        if improved:
            best_val_auc = val_m["auc"]
            no_improve   = 0
            save_best_checkpoint(
                best_ckpt, model, epoch, best_val_auc,
                cfg, feature_cols, target_col,
            )
            flag = "✅"
        else:
            no_improve += 1
            flag = ""

        # ── Resume checkpoint every epoch ─────────────────────────────
        save_resume_checkpoint(
            resume_ckpt, model, optimiser, scheduler,
            epoch, best_val_auc, no_improve,
            history, cfg, feature_cols, target_col,
        )

        # ── Print row ─────────────────────────────────────────────────
        row = ("  " +
               str(epoch).rjust(4)                        + "  " +
               str(round(train_m["loss"], 4)).rjust(7)    + "  " +
               str(round(train_m["acc"],  3)).rjust(6)    + "  " +
               str(round(train_m["auc"],  3)).rjust(6)    + "  " +
               str(round(train_m["f1"],   3)).rjust(6)    + "  " +
               str(round(val_m["loss"],   4)).rjust(7)    + "  " +
               str(round(val_m["acc"],    3)).rjust(6)    + "  " +
               str(round(val_m["auc"],    3)).rjust(6)    + "  " +
               str(round(val_m["f1"],     3)).rjust(6)    + "  " +
               str(round(current_lr,      7)).rjust(9)    + "  " +
               flag)
        tqdm.write(row)

        pbar.set_postfix({
            "val_auc": round(val_m["auc"], 4),
            "best":    round(best_val_auc, 4),
        })

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_auc=%.4f  "
            "val_loss=%.4f  val_auc=%.4f  lr=%.2e",
            epoch, epochs,
            train_m["loss"], train_m["auc"],
            val_m["loss"],   val_m["auc"],
            current_lr,
        )

        # ── Early stopping ────────────────────────────────────────────
        if no_improve >= patience:
            tqdm.write(
                "\n  Early stopping at epoch " + str(epoch) +
                " (no val AUC improvement for " + str(patience) + " epochs)."
            )
            break

    # ── Load best weights ─────────────────────────────────────────────
    load_best_checkpoint(best_ckpt, model, device)
    print("\n  Best val AUC : " + str(round(best_val_auc, 4)) +
          "  |  loaded from " + best_ckpt)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test(model, ensemble, test_loader, device):
    """
    Evaluates LSTM and optionally ensemble on the test set.

    Returns
    -------
    dict with keys: lstm, ensemble (if enabled)
    """
    criterion              = nn.BCEWithLogitsLoss()
    test_m, probs, targets = _eval_epoch(model, test_loader, criterion, device)

    print("\n-- Test Set Results --------------------------------------------------")
    print("  " + "Model".ljust(15) +
          "Acc".rjust(7) + "  " +
          "AUC".rjust(6) + "  " +
          "F1".rjust(6)  + "  " +
          "Loss".rjust(8))
    print("  " + "-" * 45)
    print("  " + "LSTM".ljust(15) +
          str(round(test_m["acc"],  3)).rjust(7) + "  " +
          str(round(test_m["auc"],  3)).rjust(6) + "  " +
          str(round(test_m["f1"],   3)).rjust(6) + "  " +
          str(round(test_m["loss"], 4)).rjust(8))

    results = {"lstm": test_m}

    if ensemble is not None:
        ens_probs = ensemble.predict_proba(test_loader, device)
        ens_m     = compute_metrics(ens_probs, targets)
        print("  " + "Ensemble".ljust(15) +
              str(round(ens_m["acc"], 3)).rjust(7) + "  " +
              str(round(ens_m["auc"], 3)).rjust(6) + "  " +
              str(round(ens_m["f1"],  3)).rjust(6) + "  " +
              "N/A".rjust(8))
        results["ensemble"] = ens_m

    print("----------------------------------------------------------------------\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(scaled_dict, feature_cols, target_col,
        cfg, train_loader, val_loader, test_loader):
    """
    Full pipeline: build -> train -> ensemble -> test evaluation.

    Parameters
    ----------
    scaled_dict  : dict  { ticker -> (train_df, val_df, test_df) }
    feature_cols : FEATURE_COLS
    target_col   : TARGET_COL
    cfg          : dict  (loaded config.yaml)
    train_loader : DataLoader  (all tickers pooled)
    val_loader   : DataLoader
    test_loader  : DataLoader

    Returns
    -------
    dict with keys:
      model        : best PSXLSTMModel
      ensemble     : TreeEnsemble or None
      history      : training history dict
      best_val_auc : float
      test_results : dict
    """
    t_cfg    = cfg.get("training", {})
    ckpt_dir = t_cfg.get("ckpt_dir",
                         "/content/psx-lstm-predictor/data/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)
    n      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PSXLSTMModel built  |  params: " + str(n) + "  |  device: " + str(device))

    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        ckpt_dir     = ckpt_dir,
        feature_cols = feature_cols,
        target_col   = target_col,
    )

    best_val_auc = max(history["val_auc"])

    ensemble = build_ensemble(cfg, model, train_loader, val_loader, device)

    test_results = evaluate_test(model, ensemble, test_loader, device)

    return {
        "model":        model,
        "ensemble":     ensemble,
        "history":      history,
        "best_val_auc": best_val_auc,
        "test_results": test_results,
    }
