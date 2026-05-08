"""
pipeline.py
=============
Single entry point that orchestrates all steps end-to-end.

Run order
---------
  Step 1  — load config.yaml
  Step 2  — download OHLCV data          (psx_downloader.py)
  Step 3  — collect + score news         (news_scraper.py)
  Step 4  — compute technical indicators (indicators.py)
  Step 5  — build scale-free features    (features.py)
  Step 6  — scale per ticker             (scaler.py)
  Step 7  — train LSTM / Transformer     (trainer.py)
  Step 8  — generate predictions         (predict.py)
  Step 9  — save plots                   (plots.py)

Usage
-----
  # Full pipeline
  python pipeline.py

  # Skip news (faster, uses zero sentiment)
  python pipeline.py --skip-news

  # Skip training (use existing checkpoints)
  python pipeline.py --skip-train

  # Only re-run evaluation + plots
  python pipeline.py --skip-news --skip-train

  # Override config path
  python pipeline.py --config my_config.yaml
"""

import os
import sys
import argparse
import logging
import time
import yaml
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# Config loader
# =============================================================================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Config loaded: %s", path)
    return cfg


# =============================================================================
# Step 2 — OHLCV download
# =============================================================================

def step_download(cfg):
    log.info("=" * 60)
    log.info("STEP 2 — Downloading PSX OHLCV data")
    log.info("=" * 60)

    from src.data_collection.psx_downloader import run as dl_run
    prices_dict = dl_run(cfg)
    log.info("Downloaded %d tickers.", len(prices_dict))
    return prices_dict


# =============================================================================
# Step 3 — News sentiment
# =============================================================================

def step_news(cfg, trading_dates):
    log.info("=" * 60)
    log.info("STEP 3 — Collecting and scoring news sentiment")
    log.info("=" * 60)

    from src.data_collection.news_scraper import run as news_run
    sentiment_df = news_run(cfg=cfg, trading_dates=trading_dates)
    log.info("Sentiment rows: %d", len(sentiment_df))
    return sentiment_df


def load_cached_sentiment(cfg):
    """
    Loads already-computed sentiment CSV if --skip-news is set.
    Falls back to None (zero sentiment) if file does not exist.
    """
    path = os.path.join(
        cfg["data"]["raw_news_dir"], "news_sentiment_daily.csv"
    )
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        log.info("Cached sentiment loaded: %s  (%d rows)", path, len(df))
        return df
    log.warning(
        "No cached sentiment found at %s — using zero sentiment.", path
    )
    return None


# =============================================================================
# Step 4 + 5 — Indicators + Features
# =============================================================================

def step_features(cfg, prices_dict, sentiment_df):
    log.info("=" * 60)
    log.info("STEP 4+5 — Computing indicators and building features")
    log.info("=" * 60)

    from src.feature_engineering.indicators import add_all_indicators
    from src.feature_engineering.features   import (
        build_features, FEATURE_COLS, TARGET_COL,
    )

    proc_dir = cfg["data"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    features_dict = {}

    for ticker, df_raw in prices_dict.items():
        log.info("  Processing: %s", ticker)

        # Normalise columns
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [c[0].lower() for c in df_raw.columns]
        else:
            df_raw.columns = [c.lower() for c in df_raw.columns]

        df_raw = df_raw[["open", "high", "low", "close", "volume"]].dropna()
        df_raw = df_raw.reset_index().rename(
            columns={"Date": "date", "index": "date"}
        )
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        df_ind  = add_all_indicators(df_raw, cfg=cfg)
        df_feat = build_features(df_ind, sentiment_df=sentiment_df, cfg=cfg)

        # Cache to disk
        out_path = os.path.join(proc_dir, f"{ticker}_features.csv")
        df_feat.to_csv(out_path, index=False)
        log.info("    Features saved -> %s  (%d rows)", out_path, len(df_feat))

        features_dict[ticker] = df_feat

    log.info("Features built for %d tickers.", len(features_dict))
    return features_dict, FEATURE_COLS, TARGET_COL


# =============================================================================
# Step 6 — Scaling
# =============================================================================

def step_scale(cfg, features_dict, feature_cols, target_col):
    log.info("=" * 60)
    log.info("STEP 6 — Scaling features (per-ticker, train-only fit)")
    log.info("=" * 60)

    from src.models.scaler import run as scale_run

    scaled_dict, scaler_dict = scale_run(
        features_dict = features_dict,
        feature_cols  = feature_cols,
        target_col    = target_col,
        out_dir       = cfg["data"]["processed_dir"],
        train_ratio   = cfg["training"].get("train_ratio", 0.70),
        val_ratio     = cfg["training"].get("val_ratio",   0.15),
    )
    return scaled_dict, scaler_dict


# =============================================================================
# Step 7 — Training
# =============================================================================

def step_train(cfg, scaled_dict, feature_cols, target_col):
    log.info("=" * 60)
    log.info("STEP 7 — Training models")
    log.info("=" * 60)

    from src.models.trainer import run as train_run

    ckpt_dir     = cfg["model"]["checkpoints_dir"]
    history_dict = train_run(
        scaled_dict  = scaled_dict,
        feature_cols = feature_cols,
        target_col   = target_col,
        cfg          = cfg,
        ckpt_dir     = ckpt_dir,
    )
    return history_dict


# =============================================================================
# Step 8 — Predictions
# =============================================================================

def step_predict(cfg, scaled_dict, scaler_dict):
    log.info("=" * 60)
    log.info("STEP 8 — Generating predictions")
    log.info("=" * 60)

    from src.evaluation.predict import run as pred_run

    pred_dict, metrics_dict = pred_run(
        scaled_dict = scaled_dict,
        scaler_dict = scaler_dict,
        cfg         = cfg,
    )
    return pred_dict, metrics_dict


# =============================================================================
# Step 9 — Plots
# =============================================================================

def step_plots(cfg, pred_dict, metrics_dict, history_dict):
    log.info("=" * 60)
    log.info("STEP 9 — Saving plots")
    log.info("=" * 60)

    from src.evaluation.plots import run as plots_run

    paths = plots_run(
        pred_dict    = pred_dict,
        metrics_dict = metrics_dict,
        history_dict = history_dict,
        cfg          = cfg,
    )
    log.info("Plots saved: %d files", len(paths))
    return paths


# =============================================================================
# Argument parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PSX LSTM Predictor — end-to-end pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML  (default: config.yaml)"
    )
    parser.add_argument(
        "--skip-news", action="store_true",
        help="Skip news collection; load cached CSV or use zero sentiment"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; load existing checkpoints for inference"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation"
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Logging setup ──────────────────────────────────────────────────
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a"),
        ],
    )

    t0 = time.time()
    log.info("╔══════════════════════════════════════╗")
    log.info("║   PSX LSTM Predictor — Pipeline      ║")
    log.info("╚══════════════════════════════════════╝")

    # ── Step 1: Config ─────────────────────────────────────────────────
    cfg = load_config(args.config)

    # Create all output directories upfront
    for key in ("raw_prices_dir", "raw_news_dir",
                "processed_dir", "predictions_dir"):
        os.makedirs(cfg["data"].get(key, f"data/{key}"), exist_ok=True)
    os.makedirs(cfg["model"]["checkpoints_dir"], exist_ok=True)

    # ── Step 2: Download ───────────────────────────────────────────────
    prices_dict = step_download(cfg)

    # Collect all trading dates across tickers for news alignment
    all_dates = sorted(set(
        d for df in prices_dict.values()
        for d in pd.to_datetime(
            df.reset_index()["Date"]
            if "Date" in df.reset_index().columns
            else df.index
        )
    ))
    trading_dates = pd.DatetimeIndex(all_dates)

    # ── Step 3: News ───────────────────────────────────────────────────
    if args.skip_news:
        log.info("STEP 3 — Skipped (--skip-news)")
        sentiment_df = load_cached_sentiment(cfg)
    else:
        sentiment_df = step_news(cfg, trading_dates)

    # ── Step 4+5: Features ─────────────────────────────────────────────
    features_dict, feature_cols, target_col = step_features(
        cfg, prices_dict, sentiment_df
    )

    # ── Step 6: Scale ──────────────────────────────────────────────────
    scaled_dict, scaler_dict = step_scale(
        cfg, features_dict, feature_cols, target_col
    )

    # ── Step 7: Train ──────────────────────────────────────────────────
    if args.skip_train:
        log.info("STEP 7 — Skipped (--skip-train)")
        history_dict = {}
    else:
        history_dict = step_train(
            cfg, scaled_dict, feature_cols, target_col
        )

    # ── Step 8: Predict ────────────────────────────────────────────────
    pred_dict, metrics_dict = step_predict(cfg, scaled_dict, scaler_dict)

    # ── Step 9: Plots ──────────────────────────────────────────────────
    if args.skip_plots:
        log.info("STEP 9 — Skipped (--skip-plots)")
    else:
        step_plots(cfg, pred_dict, metrics_dict, history_dict)

    # ── Done ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Pipeline complete in %.1f seconds.", elapsed)
    log.info("=" * 60)

    # Final metrics summary
    print("\n=== Final Test Metrics ===")
    for ticker, m in metrics_dict.items():
        print(
            f"  {ticker:12s}  "
            f"MAE={m.get('MAE', 0):.5f}  "
            f"RMSE={m.get('RMSE', 0):.5f}  "
            f"DirAcc={m.get('DirectionalAcc_pct', 0):.1f}%"
        )


if __name__ == "__main__":
    main()
