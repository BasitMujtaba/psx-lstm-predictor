"""
src/evaluation/plots.py
=========================
Visualisation functions for PSX LSTM prediction results.

Plots produced
--------------
  1. prediction_vs_actual   — predicted vs actual returns over time
  2. scatter_plot           — scatter of pred vs actual with R² annotation
  3. residuals_plot         — residuals (pred - actual) over time
  4. training_history       — train/val loss curves per ticker
  5. directional_accuracy   — bar chart of directional accuracy per ticker
  6. all_tickers_summary    — combined metrics bar chart across tickers

All plots are saved to data/predictions/<TICKER>_<plot_name>.png
and optionally displayed inline (show=True for Colab / Jupyter).

config.yaml section used
------------------------
  data:
    predictions_dir: data/predictions
  plots:
    dpi:        150
    figsize:    [14, 5]
    show:       false       # set true in Colab to display inline
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

log = logging.getLogger(__name__)

# Consistent colour palette across all plots
COLOURS = {
    "actual":    "#2196F3",   # blue
    "pred":      "#FF5722",   # deep orange
    "residual":  "#9C27B0",   # purple
    "train":     "#4CAF50",   # green
    "val":       "#FF9800",   # amber
    "bar":       "#00BCD4",   # cyan
    "grid":      "#E0E0E0",
    "zero":      "#B0BEC5",
}


# =============================================================================
# Helpers
# =============================================================================

def _savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Plot saved -> %s", path)


def _apply_grid(ax):
    ax.grid(True, color=COLOURS["grid"], linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)


def _date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


# =============================================================================
# 1. Prediction vs Actual  (line plot)
# =============================================================================

def plot_prediction_vs_actual(pred_df, ticker, out_dir,
                               split="test", dpi=150,
                               figsize=(14, 5), show=False):
    """
    Line chart of predicted vs actual daily % return for one split.
    """
    df = pred_df[pred_df["split"] == split].copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df["date"], df["actual_pct"],
            color=COLOURS["actual"], linewidth=1.2,
            label="Actual return %", alpha=0.85)
    ax.plot(df["date"], df["pred_pct"],
            color=COLOURS["pred"],   linewidth=1.2,
            label="Predicted return %", alpha=0.85, linestyle="--")

    ax.axhline(0, color=COLOURS["zero"], linewidth=0.8, linestyle=":")
    _apply_grid(ax)
    _date_axis(ax)

    ax.set_title(f"{ticker} — Predicted vs Actual Return ({split})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_prediction_vs_actual.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 2. Scatter plot  (pred vs actual)
# =============================================================================

def plot_scatter(pred_df, ticker, out_dir,
                 split="test", dpi=150,
                 figsize=(6, 6), show=False):
    """
    Scatter plot of predicted vs actual returns with R² annotation.
    """
    df = pred_df[pred_df["split"] == split].copy()

    actual = df["actual_pct"].values
    pred   = df["pred_pct"].values

    # R²
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(actual, pred, alpha=0.4, s=12,
               color=COLOURS["pred"], edgecolors="none")

    # Perfect prediction line
    lim = max(abs(actual).max(), abs(pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim],
            color=COLOURS["actual"], linewidth=1.2,
            linestyle="--", label="Perfect prediction")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _apply_grid(ax)

    ax.set_title(f"{ticker} — Scatter ({split})  R²={r2:.3f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Return (%)")
    ax.set_ylabel("Predicted Return (%)")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_scatter.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 3. Residuals plot
# =============================================================================

def plot_residuals(pred_df, ticker, out_dir,
                   split="test", dpi=150,
                   figsize=(14, 4), show=False):
    """
    Residuals (pred - actual) over time with a zero line.
    """
    df = pred_df[pred_df["split"] == split].copy()
    df["date"]     = pd.to_datetime(df["date"])
    df["residual"] = df["pred_pct"] - df["actual_pct"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(df["date"], df["residual"],
           color=COLOURS["residual"], alpha=0.55, width=1.5)
    ax.axhline(0, color=COLOURS["zero"], linewidth=1.0)
    _apply_grid(ax)
    _date_axis(ax)

    ax.set_title(f"{ticker} — Residuals  pred − actual ({split})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_residuals.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 4. Training history  (loss curves)
# =============================================================================

def plot_training_history(history, ticker, out_dir,
                          dpi=150, figsize=(10, 4), show=False):
    """
    Plots train and val loss curves for one ticker.

    Parameters
    ----------
    history : dict with keys "train_loss", "val_loss"
              (output of trainer.train)
    """
    epochs     = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epochs, train_loss,
            color=COLOURS["train"], linewidth=1.5, label="Train loss")
    ax.plot(epochs, val_loss,
            color=COLOURS["val"],   linewidth=1.5,
            label="Val loss", linestyle="--")

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val   = min(val_loss)
    ax.axvline(best_epoch, color=COLOURS["zero"],
               linewidth=1.0, linestyle=":", label=f"Best epoch ({best_epoch})")

    _apply_grid(ax)
    ax.set_title(f"{ticker} — Training History",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.legend(framealpha=0.9)
    ax.annotate(f"val={best_val:.5f}",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1, best_val * 1.05),
                fontsize=8, color=COLOURS["val"])

    fig.tight_layout()
    path = os.path.join(out_dir, f"{ticker}_training_history.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 5. Directional accuracy bar chart  (per ticker)
# =============================================================================

def plot_directional_accuracy(metrics_dict, out_dir,
                               dpi=150, figsize=(10, 5), show=False):
    """
    Horizontal bar chart comparing directional accuracy across tickers.

    Parameters
    ----------
    metrics_dict : { ticker -> metrics dict }  output of predict.run()
    """
    tickers = list(metrics_dict.keys())
    accs    = [metrics_dict[t].get("DirectionalAcc_pct", 0) for t in tickers]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(tickers, accs,
                   color=COLOURS["bar"], alpha=0.8, edgecolor="white")

    ax.axvline(50, color=COLOURS["zero"],
               linewidth=1.2, linestyle="--", label="Random baseline (50%)")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=9)

    _apply_grid(ax)
    ax.set_xlim(0, 105)
    ax.set_title("Directional Accuracy by Ticker (Test Split)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Directional Accuracy (%)")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, "all_directional_accuracy.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 6. All-tickers summary  (MAE + RMSE grouped bar)
# =============================================================================

def plot_metrics_summary(metrics_dict, out_dir,
                         dpi=150, figsize=(12, 5), show=False):
    """
    Grouped bar chart of MAE and RMSE for every ticker on the test split.
    """
    tickers = list(metrics_dict.keys())
    mae     = [metrics_dict[t].get("MAE",  0) for t in tickers]
    rmse    = [metrics_dict[t].get("RMSE", 0) for t in tickers]

    x     = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width / 2, mae,  width,
           label="MAE",  color=COLOURS["actual"], alpha=0.8)
    ax.bar(x + width / 2, rmse, width,
           label="RMSE", color=COLOURS["pred"],   alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    _apply_grid(ax)

    ax.set_title("MAE & RMSE by Ticker (Test Split — log-return scale)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Error (log-return scale)")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, "all_metrics_summary.png")
    _savefig(fig, path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# Run for all tickers  (called from pipeline.py)
# =============================================================================

def run(pred_dict, metrics_dict, history_dict, cfg):
    """
    Generates all plots for every ticker.

    Parameters
    ----------
    pred_dict    : { ticker -> prediction DataFrame }
    metrics_dict : { ticker -> metrics dict }
    history_dict : { ticker -> training history dict }
    cfg          : full config dict

    Returns
    -------
    list of all saved plot paths
    """
    out_dir  = cfg["data"]["predictions_dir"]
    p_cfg    = cfg.get("plots", {})
    dpi      = p_cfg.get("dpi",     150)
    figsize  = tuple(p_cfg.get("figsize", [14, 5]))
    show     = p_cfg.get("show",   False)

    all_paths = []

    for ticker, pred_df in pred_dict.items():
        log.info("Plotting: %s", ticker)

        all_paths.append(plot_prediction_vs_actual(
            pred_df, ticker, out_dir,
            split="test", dpi=dpi, figsize=figsize, show=show,
        ))
        all_paths.append(plot_scatter(
            pred_df, ticker, out_dir,
            split="test", dpi=dpi, show=show,
        ))
        all_paths.append(plot_residuals(
            pred_df, ticker, out_dir,
            split="test", dpi=dpi, figsize=figsize, show=show,
        ))
        if ticker in history_dict:
            all_paths.append(plot_training_history(
                history_dict[ticker], ticker, out_dir,
                dpi=dpi, show=show,
            ))

    # Cross-ticker summary plots
    if metrics_dict:
        all_paths.append(plot_directional_accuracy(
            metrics_dict, out_dir, dpi=dpi, show=show,
        ))
        all_paths.append(plot_metrics_summary(
            metrics_dict, out_dir, dpi=dpi, show=show,
        ))

    log.info("All plots saved: %d files", len(all_paths))
    return all_paths


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Synthetic prediction DataFrame
    np.random.seed(42)
    n   = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    pred_df = pd.DataFrame({
        "date":          dates,
        "close":         np.cumprod(1 + np.random.randn(n) * 0.01) * 100,
        "actual_return": np.random.randn(n) * 0.01,
        "pred_return":   np.random.randn(n) * 0.01,
        "actual_pct":    np.random.randn(n) * 1.0,
        "pred_pct":      np.random.randn(n) * 1.0,
        "split":         ["test"] * n,
    })

    history = {
        "train_loss": list(np.exp(-np.linspace(0, 2, 30)) + np.random.rand(30) * 0.02),
        "val_loss":   list(np.exp(-np.linspace(0, 1.8, 30)) + np.random.rand(30) * 0.03),
    }

    metrics_dict = {
        "OGDC.KA": {"MAE": 0.0045, "RMSE": 0.0062, "DirectionalAcc_pct": 54.3},
        "LUCK.KA": {"MAE": 0.0051, "RMSE": 0.0071, "DirectionalAcc_pct": 52.1},
    }

    OUT = "/tmp/smoke_plots"

    plot_prediction_vs_actual(pred_df, "OGDC.KA", OUT, show=False)
    plot_scatter(pred_df,   "OGDC.KA", OUT, show=False)
    plot_residuals(pred_df, "OGDC.KA", OUT, show=False)
    plot_training_history(history, "OGDC.KA", OUT, show=False)
    plot_directional_accuracy(metrics_dict, OUT, show=False)
    plot_metrics_summary(metrics_dict,      OUT, show=False)

    print("\n✓ plots.py smoke-test passed.")
    print(f"  Plots saved to: {OUT}")
