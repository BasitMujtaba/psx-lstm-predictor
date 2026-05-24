"""
src/models/trainer.py
=======================
Training loop with early stopping, LR scheduling, epoch-by-epoch
checkpointing, and automatic resume if session breaks.

What this module does
---------------------
  1. Trains PSXLSTMModel (binary classification UP=1/DOWN=0)
  2. BCEWithLogitsLoss with label smoothing + pos_weight
  3. Evaluates loss, accuracy, AUC, F1 on val set every epoch
  4. Saves best checkpoint (highest val AUC) to disk
  5. Saves resume checkpoint after EVERY epoch
  6. Automatically resumes from last completed epoch if session breaks
  7. Stops early if val AUC does not improve for patience epochs
  8. AdamW + CosineAnnealingLR + linear warmup
  9. Gradient clipping

Saved artefacts
---------------
  checkpoints/best_model.pt   -- best val AUC weights
  checkpoints/resume_model.pt -- saved every epoch for resume

config.yaml sections used
-------------------------
  model:
    lstm_dropout:    0.6
    checkpoints_dir: src/models/checkpoints

  training:
    max_epochs:    50
    patience:      10
    lr:            0.0003
    weight_decay:  0.005
    grad_clip:     1.0
    warmup_epochs: 3
    batch_size:    256
    label_smooth:  0.1
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


# -- 1. Metrics ----------------------------------------------------------------

def compute_metrics(probs, targets, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "acc": accuracy_score(targets, preds),
        "auc": roc_auc_score(targets, probs),
        "f1":  f1_score(targets, preds, zero_division=0),
    }


# -- 2. Checkpoint Helpers -----------------------------------------------------

def save_best_checkpoint(path, model, epoch, val_auc,
                         cfg, feature_cols, target_col):
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
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    log.info("Best checkpoint loaded <- %s  (epoch=%d  val_auc=%.4f)",
             path, ckpt["epoch"], ckpt["val_auc"])
    return ckpt


# -- 3. LR Warmup --------------------------------------------------------------

def _warmup_lr(optimiser, epoch, warmup_epochs, base_lr):
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
        for pg in optimiser.param_groups:
            pg["lr"] = max(lr, 1e-7)


# -- 4. One Epoch Train --------------------------------------------------------

def _train_epoch(model, loader, criterion, optimiser,
                 device, grad_clip, label_smooth=0.1):
    model.train()
    total_loss = 0.0
    all_probs, all_targets = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # label smoothing: 0 -> 0.05, 1 -> 0.95
        yb_smooth = yb * (1 - label_smooth) + 0.5 * label_smooth

        optimiser.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb_smooth)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()

        total_loss += loss.item() * len(yb)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(yb.cpu().numpy())   # original labels for metrics

    probs           = np.concatenate(all_probs)
    targets         = np.concatenate(all_targets)
    metrics         = compute_metrics(probs, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# -- 5. One Epoch Evaluate -----------------------------------------------------

@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits  = model(xb)
        loss    = criterion(logits, yb)   # no smoothing on val/test

        total_loss += loss.item() * len(yb)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs           = np.concatenate(all_probs)
    targets         = np.concatenate(all_targets)
    metrics         = compute_metrics(probs, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, probs, targets


# -- 6. Full Training Loop -----------------------------------------------------

def train(model, train_loader, val_loader,
          cfg, ckpt_dir, feature_cols, target_col):
    t_cfg         = cfg.get("training", {})
    epochs        = t_cfg.get("max_epochs",    50)
    patience      = t_cfg.get("patience",      10)
    base_lr       = t_cfg.get("lr",        0.0003)
    weight_decay  = t_cfg.get("weight_decay", 0.005)
    grad_clip     = t_cfg.get("grad_clip",     1.0)
    warmup_epochs = t_cfg.get("warmup_epochs",   3)
    label_smooth  = t_cfg.get("label_smooth",  0.1)

    best_ckpt   = os.path.join(ckpt_dir, "best_model.pt")
    resume_ckpt = os.path.join(ckpt_dir, "resume_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_pos      = sum((b[1] == 1).sum().item() for b in train_loader)
    n_neg      = sum((b[1] == 0).sum().item() for b in train_loader)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log.info("pos_weight=%.4f  (n_pos=%d  n_neg=%d)", pos_weight.item(), n_pos, n_neg)

    optimiser = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {
        "train_loss": [], "train_acc": [], "train_auc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [], "val_f1":   [],
        "lr":         [],
    }
    best_val_auc = 0.0
    no_improve   = 0
    start_epoch  = 1

    if os.path.exists(resume_ckpt):
        print("  Resume checkpoint found: " + resume_ckpt)
        start_epoch, best_val_auc, no_improve, history = load_resume_checkpoint(
            resume_ckpt, model, optimiser, scheduler, device,
        )
        start_epoch += 1
    else:
        print("  No resume checkpoint -- starting from scratch.")

    print("")
    print("  " +
          "Ep".rjust(4)     + "  " +
          "TrLoss".rjust(7) + "  " +
          "TrAcc".rjust(6)  + "  " +
          "TrAUC".rjust(6)  + "  " +
          "TrF1".rjust(6)   + "  " +
          "VlLoss".rjust(7) + "  " +
          "VlAcc".rjust(6)  + "  " +
          "VlAUC".rjust(6)  + "  " +
          "VlF1".rjust(6)   + "  " +
          "LR".rjust(9))
    print("  " + "-" * 90)

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:

        if epoch <= warmup_epochs:
            _warmup_lr(optimiser, epoch, warmup_epochs, base_lr)

        train_m     = _train_epoch(model, train_loader, criterion,
                                   optimiser, device, grad_clip, label_smooth)
        val_m, _, _ = _eval_epoch(model, val_loader, criterion, device)
        current_lr  = optimiser.param_groups[0]["lr"]

        if epoch > warmup_epochs:
            scheduler.step()

        for k in ["loss", "acc", "auc", "f1"]:
            history["train_" + k].append(train_m[k])
            history["val_"   + k].append(val_m[k])
        history["lr"].append(current_lr)

        improved = val_m["auc"] > best_val_auc
        if improved:
            best_val_auc = val_m["auc"]
            no_improve   = 0
            save_best_checkpoint(
                best_ckpt, model, epoch, best_val_auc,
                cfg, feature_cols, target_col,
            )
            flag = "best"
        else:
            no_improve += 1
            flag = ""

        save_resume_checkpoint(
            resume_ckpt, model, optimiser, scheduler,
            epoch, best_val_auc, no_improve,
            history, cfg, feature_cols, target_col,
        )

        tqdm.write(
            "  " + str(epoch).rjust(4)                 + "  " +
            str(round(train_m["loss"], 4)).rjust(7)     + "  " +
            str(round(train_m["acc"],  3)).rjust(6)     + "  " +
            str(round(train_m["auc"],  3)).rjust(6)     + "  " +
            str(round(train_m["f1"],   3)).rjust(6)     + "  " +
            str(round(val_m["loss"],   4)).rjust(7)     + "  " +
            str(round(val_m["acc"],    3)).rjust(6)     + "  " +
            str(round(val_m["auc"],    3)).rjust(6)     + "  " +
            str(round(val_m["f1"],     3)).rjust(6)     + "  " +
            str(round(current_lr,      7)).rjust(9)     + "  " +
            flag
        )

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

        if no_improve >= patience:
            tqdm.write(
                "  Early stopping at epoch " + str(epoch) +
                "  (no improvement for " + str(patience) + " epochs)"
            )
            break

    load_best_checkpoint(best_ckpt, model, device)
    print("  Best val AUC : " + str(round(best_val_auc, 4)) +
          "  |  loaded from " + best_ckpt)
    return history


# -- 7. Test Evaluation --------------------------------------------------------

def evaluate_test(model, ensemble, test_loader, device):
    criterion              = nn.BCEWithLogitsLoss()
    test_m, probs, targets = _eval_epoch(model, test_loader, criterion, device)

    print("")
    print("-- Test Results ------------------------------------------------------")
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

    print("---------------------------------------------------------------------")
    return results


# -- 8. Main Entry Point -------------------------------------------------------

def run(scaled_dict, feature_cols, target_col,
        cfg, train_loader, val_loader, test_loader):
    m_cfg    = cfg.get("model", {})
    ckpt_dir = m_cfg.get("checkpoints_dir",
                         "/content/psx-lstm-predictor/src/models/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)
    n      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PSXLSTMModel  |  params: " + str(n) + "  |  device: " + str(device))

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
    ensemble     = build_ensemble(cfg, model, train_loader, val_loader, device)
    test_results = evaluate_test(model, ensemble, test_loader, device)

    return {
        "model":        model,
        "ensemble":     ensemble,
        "history":      history,
        "best_val_auc": best_val_auc,
        "test_results": test_results,
    }


if __name__ == "__main__":
    run()
