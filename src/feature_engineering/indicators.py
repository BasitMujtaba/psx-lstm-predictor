"""
================================================================================
 File   : src/feature_engineering/indicators.py
 Project: PSX LSTM Predictor
 Purpose: Computes technical indicators on raw PSX OHLCV data and saves
          the processed file with warmup rows dropped.

 Input  : data/raw/psx_prices/psx_prices_raw.csv
 Output : data/processed/psx_prices_processed.csv

 Indicators computed per ticker:
   MACD, RSI(14), CCI(20), DMI DX(14)
   EMA(9, 21, 50, 200)
   Bollinger Bands (mid, upper, lower, width, pct)
   Turbulence (rolling 252-day)

 Cache logic:
   1. Processed CSV exists and is valid -> return directly
   2. Processed CSV missing             -> load raw, compute, save, push
================================================================================
"""

import os, logging, subprocess
import numpy as np
import pandas as pd
import yaml
from tqdm.notebook import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EMA_PERIODS = [9, 21, 50, 200]
_WARMUP_ROWS = 252


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


def _processed_cache_valid(path):
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return False, None
        log.info("Processed cache hit — %s -> %s (%d rows, %d tickers)",
                 df["date"].min().date(), df["date"].max().date(),
                 len(df), df["ticker"].nunique())
        df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        if not df.equals(df_sorted):
            log.info("Sort order wrong — fixing and re-saving ...")
            df_sorted.to_csv(path, index=False)
            return True, df_sorted
        return True, df
    except Exception as e:
        log.warning("Processed cache check failed: %s", e)
        return False, None


def _push_to_github(processed_path, raw_path, start, end):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "pull", "--rebase", "origin", "main"],
            ["git", "-C", PROJECT_ROOT, "add", raw_path, processed_path],
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             f"Update PSX prices cache {start} -> {end}"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed processed CSV to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu, sig = np.mean(window[:-1]), np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def _drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = (df.groupby("ticker", group_keys=False)
            .apply(lambda x: x.iloc[_WARMUP_ROWS:])
            .reset_index(drop=True))
    after_warmup = len(df)
    df = df.dropna().reset_index(drop=True)
    after_nan = len(df)
    log.info(
        "Warmup drop: %d -> %d rows (-%d warmup) -> %d rows (-%d NaN)",
        before, after_warmup, before - after_warmup,
        after_nan, after_warmup - after_nan,
    )
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing technical indicators for %d tickers ...", df["ticker"].nunique())
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    frames = []
    for ticker, grp in tqdm(df.groupby("ticker"), desc="Indicators"):
        g                = grp.copy().sort_values("date").reset_index(drop=True)
        close, high, low = g["close"], g["high"], g["low"]

        # ── MACD
        g["macd"] = (close.ewm(span=12, adjust=False).mean()
                     - close.ewm(span=26, adjust=False).mean())

        # ── RSI(14)
        delta    = close.diff()
        rs       = (delta.clip(lower=0).rolling(14).mean()
                    / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
        g["rsi"] = 100 - (100 / (1 + rs))

        # ── CCI(20)
        tp       = (high + low + close) / 3
        mad      = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        g["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

        # ── DMI DX(14)
        prev_close = close.shift(1)
        up_move    = high - high.shift(1)
        dn_move    = low.shift(1) - low
        pos_dm     = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm     = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr         = pd.concat([high - low,
                                 (high - prev_close).abs(),
                                 (low  - prev_close).abs()], axis=1).max(axis=1)
        atr14      = tr.rolling(14).mean()
        pdi14      = (100 * pd.Series(pos_dm, index=g.index).rolling(14).mean()
                      / atr14.replace(0, np.nan))
        ndi14      = (100 * pd.Series(neg_dm, index=g.index).rolling(14).mean()
                      / atr14.replace(0, np.nan))
        g["dmi_dx"] = (100 * (pdi14 - ndi14).abs()
                       / (pdi14 + ndi14).replace(0, np.nan))

        # ── EMA(9, 21, 50, 200)
        for period in _EMA_PERIODS:
            g[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # ── Bollinger Bands(20)
        bb_sma        = close.rolling(20).mean()
        bb_std        = close.rolling(20).std(ddof=0)
        g["bb_mid"]   = bb_sma
        g["bb_upper"] = bb_sma + 2 * bb_std
        g["bb_lower"] = bb_sma - 2 * bb_std
        bb_range      = (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
        g["bb_width"] = bb_range / g["bb_mid"].replace(0, np.nan)
        g["bb_pct"]   = (close - g["bb_lower"]) / bb_range

        # ── Turbulence (rolling 252-day)
        g["turbulence"] = (close.pct_change()
                               .rolling(252)
                               .apply(_turbulence_1d, raw=True)
                               .fillna(0.0))
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result = _drop_warmup_rows(result)
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    indicator_cols = (["macd", "rsi", "cci", "dmi_dx"]
                      + [f"ema_{p}" for p in _EMA_PERIODS]
                      + ["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct", "turbulence"])
    nan_report = result[indicator_cols].isna().mean().mul(100).round(1).to_string()
    log.info("NaN per indicator (should all be 0.0 after warmup drop):\n%s", nan_report)
    return result


def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    raw_dir        = _resolve(cfg["data"]["raw_prices_dir"])
    processed_dir  = _resolve(cfg["data"]["processed_dir"])
    raw_path       = os.path.join(raw_dir,       "psx_prices_raw.csv")
    processed_path = os.path.join(processed_dir, "psx_prices_processed.csv")
    start          = cfg["data"]["start_date"]
    end            = cfg["data"]["end_date"]

    os.makedirs(processed_dir, exist_ok=True)

    # ── Check processed cache first
    is_valid, cached_df = _processed_cache_valid(processed_path)
    if is_valid:
        log.info("Processed cache valid — returning directly")
        return cached_df

    # ── Load raw
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            "Run psx_scraper.run() first to fetch and save raw data."
        )
    log.info("Loading raw data from %s ...", raw_path)
    raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    log.info("Raw loaded: %d rows | %d tickers | %s -> %s",
             len(raw_df), raw_df["ticker"].nunique(),
             raw_df["date"].min().date(), raw_df["date"].max().date())

    # ── Compute indicators
    processed_df = add_technical_indicators(raw_df)

    # ── Save
    processed_df.to_csv(processed_path, index=False)
    log.info("Processed saved -> %s  (%d rows, %.1f MB)",
             processed_path, len(processed_df), os.path.getsize(processed_path) / 1e6)

    _push_to_github(processed_path, raw_path, start, end)
    return processed_df


if __name__ == "__main__":
    run()
