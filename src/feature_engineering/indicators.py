"""
================================================================================
 File   : src/feature_engineering/indicators.py
 Project: PSX LSTM Predictor
 Purpose: Loads raw OHLCV CSV, cleans dirty rows, computes technical indicators,
          drops warmup rows, saves processed CSV, then splits into per-category
          CSVs (macro / energy / banking / forex / corporate / unassigned).

 Input:
   data/raw/psx_prices/psx_prices_raw.csv

 Saves:
   data/processed/psx_prices_processed.csv
   data/processed/psx_prices_macro.csv
   data/processed/psx_prices_energy.csv
   data/processed/psx_prices_banking.csv
   data/processed/psx_prices_forex.csv
   data/processed/psx_prices_corporate.csv
   data/processed/psx_prices_unassigned.csv

 Cache logic:
   1. If processed CSV exists and is valid  -> return it directly
   2. If not                                -> compute from raw and save
================================================================================
"""

import os, logging, warnings
import numpy as np
import pandas as pd
import yaml
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EMA_PERIODS = [9, 21, 50, 200]
_WARMUP_ROWS = 252

# ── Ticker → Category mapping ─────────────────────────────────────────────────
CATEGORY_TICKERS = {
    "macro"     : ["DGKC", "EFERT", "FFC", "FATIMA"],
    "energy"    : ["HUBC", "KAPCO", "MARI", "OGDC", "POL", "PPL", "PSO"],
    "banking"   : ["BAFL", "HBL", "MCB", "NBP", "UBL"],
    "forex"     : ["SYS", "TRG", "SEARL", "FEROZ", "NML", "NCL"],
    "corporate" : ["AVN", "ENGRO", "GATM", "INDU", "LUCK", "MLCF", "PIOC", "PSMC"],
}
# ─────────────────────────────────────────────────────────────────────────────


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
        log.info("Processed cache hit: %d rows | %d tickers | %s -> %s",
                 len(df), df["ticker"].nunique(),
                 df["date"].min().date(), df["date"].max().date())
        return True, df
    except Exception as e:
        log.warning("Processed cache check failed: %s", e)
        return False, None


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["open"]   > 0].copy()
    df = df[df["volume"] > 0].copy()
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    log.info("Cleaning: removed %d dirty rows (zero open or zero volume) | %d rows remaining",
             removed, len(df))
    return df


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


def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu, sig = np.mean(window[:-1]), np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing indicators for %d tickers ...", df["ticker"].nunique())
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    frames = []
    for ticker, grp in tqdm(df.groupby("ticker"), desc="Indicators"):
        g = grp.copy().sort_values("date").reset_index(drop=True)
        close, high, low = g["close"], g["high"], g["low"]

        g["macd"] = (close.ewm(span=12, adjust=False).mean()
                     - close.ewm(span=26, adjust=False).mean())

        delta    = close.diff()
        rs       = (delta.clip(lower=0).rolling(14).mean()
                    / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
        g["rsi"] = 100 - (100 / (1 + rs))

        tp       = (high + low + close) / 3
        mad      = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        g["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

        prev_close = close.shift(1)
        up_move    = high - high.shift(1)
        dn_move    = low.shift(1) - low
        pos_dm     = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm     = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr         = pd.concat([high - low,
                                 (high - prev_close).abs(),
                                 (low  - prev_close).abs()], axis=1).max(axis=1)
        atr14      = tr.rolling(14).mean()
        pdi14      = 100 * pd.Series(pos_dm, index=g.index).rolling(14).mean() / atr14.replace(0, np.nan)
        ndi14      = 100 * pd.Series(neg_dm, index=g.index).rolling(14).mean() / atr14.replace(0, np.nan)
        g["dmi_dx"] = 100 * (pdi14 - ndi14).abs() / (pdi14 + ndi14).replace(0, np.nan)

        for period in _EMA_PERIODS:
            g[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        bb_sma        = close.rolling(20).mean()
        bb_std        = close.rolling(20).std(ddof=0)
        g["bb_mid"]   = bb_sma
        g["bb_upper"] = bb_sma + 2 * bb_std
        g["bb_lower"] = bb_sma - 2 * bb_std
        bb_range      = (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
        g["bb_width"] = bb_range / g["bb_mid"].replace(0, np.nan)
        g["bb_pct"]   = (close - g["bb_lower"]) / bb_range

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
    log.info("NaN %% per indicator (all should be 0.0):\n%s", nan_report)
    return result


def save_category_csvs(df: pd.DataFrame, processed_dir: str) -> None:
    ticker_to_cat = {}
    for cat, tickers in CATEGORY_TICKERS.items():
        for t in tickers:
            ticker_to_cat[t.upper()] = cat

    all_tickers    = set(df["ticker"].str.upper().unique())
    mapped_tickers = set(ticker_to_cat.keys())
    unassigned     = sorted(all_tickers - mapped_tickers)

    if unassigned:
        log.warning("%d tickers not in any category -> psx_prices_unassigned.csv: %s",
                    len(unassigned), unassigned)
    else:
        log.info("All %d tickers successfully mapped to a category.", len(all_tickers))

    df = df.copy()
    df["_cat"] = df["ticker"].str.upper().map(ticker_to_cat).fillna("unassigned")

    categories = list(CATEGORY_TICKERS.keys()) + (["unassigned"] if unassigned else [])

    log.info("\n── Category CSV summary ─────────────────────────────────────")
    for cat in categories:
        cat_df = (df[df["_cat"] == cat]
                  .drop(columns=["_cat"])
                  .sort_values(["ticker", "date"])
                  .reset_index(drop=True))

        out_path = os.path.join(processed_dir, f"psx_prices_{cat}.csv")
        cat_df.to_csv(out_path, index=False)
        log.info("  %-12s  %6d rows  %3d tickers  -> %s",
                 cat, len(cat_df), cat_df["ticker"].nunique(),
                 os.path.basename(out_path))

    log.info("────────────────────────────────────────────────────────────")


def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    raw_dir        = _resolve(cfg["data"]["raw_prices_dir"])
    processed_dir  = _resolve(cfg["data"]["processed_dir"])
    raw_path       = os.path.join(raw_dir,       "psx_prices_raw.csv")
    processed_path = os.path.join(processed_dir, "psx_prices_processed.csv")

    os.makedirs(processed_dir, exist_ok=True)

    is_valid, cached_df = _processed_cache_valid(processed_path)
    if is_valid:
        log.info("Returning cached processed data — regenerating category CSVs ...")
        save_category_csvs(cached_df, processed_dir)
        return cached_df

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw CSV not found: {raw_path} — run psx_downloader first")

    log.info("Loading raw CSV from %s ...", raw_path)
    raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    log.info("Raw loaded: %d rows | %d tickers", len(raw_df), raw_df["ticker"].nunique())

    raw_df = clean_raw(raw_df)

    processed_df = add_technical_indicators(raw_df)

    processed_df.to_csv(processed_path, index=False)
    log.info("Processed saved -> %s  (%d rows, %.1f MB)",
             processed_path, len(processed_df), os.path.getsize(processed_path) / 1e6)

    save_category_csvs(processed_df, processed_dir)

    return processed_df


if __name__ == "__main__":
    run()
