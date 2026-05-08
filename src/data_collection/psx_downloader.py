"""
src/data_collection/psx_downloader.py
=======================================
Fetches OHLCV price data for PSX tickers via yfinance
and computes Step 1 technical indicators.
"""

import os, time, logging, warnings
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_psx_prices(tickers, start, end, sleep=0.3):
    log.info("Fetching PSX price data for %d tickers", len(tickers))
    valid_rows, failed = [], []

    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                failed.append(ticker)
                continue

            raw = raw.reset_index()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            needed = ["date", "open", "high", "low", "close", "volume"]
            raw    = raw[[c for c in needed if c in raw.columns]].copy()
            raw["ticker"] = ticker
            raw["date"]   = pd.to_datetime(raw["date"]).dt.normalize()
            valid_rows.append(raw)
            time.sleep(sleep)

        except Exception as exc:
            log.error("Failed %s: %s", ticker, exc)
            failed.append(ticker)

    if not valid_rows:
        raise RuntimeError("No price data fetched.")

    df = pd.concat(valid_rows, ignore_index=True)
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Price data shape: %s | Tickers loaded: %d",
             df.shape, df["ticker"].nunique())
    return df, failed


def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu  = np.mean(window[:-1])
    sig = np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_technical_indicators(df):
    log.info("Computing technical indicators")
    out_frames = []

    for ticker, grp in tqdm(df.groupby("ticker"), desc="Indicators"):
        grp   = grp.copy().sort_values("date").reset_index(drop=True)
        close = grp["close"]
        high  = grp["high"]
        low   = grp["low"]

        # MACD
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        grp["macd"] = ema12 - ema26

        # RSI
        delta       = close.diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        rs          = gain / loss.replace(0, np.nan)
        grp["rsi"]  = 100 - (100 / (1 + rs))

        # CCI
        tp          = (high + low + close) / 3
        mad         = tp.rolling(20).apply(
                          lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        grp["cci"]  = (tp - tp.rolling(20).mean()) / (
                          0.015 * mad.replace(0, np.nan))

        # DMI / DX
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)
        up_move    = high - prev_high
        dn_move    = prev_low - low
        pos_dm     = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm     = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr         = pd.concat([
                         high - low,
                         (high - prev_close).abs(),
                         (low  - prev_close).abs()
                     ], axis=1).max(axis=1)
        atr14            = tr.rolling(14).mean()
        pdi14            = 100 * pd.Series(pos_dm).rolling(14).mean() / atr14.replace(0, np.nan)
        ndi14            = 100 * pd.Series(neg_dm).rolling(14).mean() / atr14.replace(0, np.nan)
        grp["dmi_dx"]    = 100 * (pdi14 - ndi14).abs() / (
                               pdi14 + ndi14).replace(0, np.nan)

        # Turbulence (needs 252 days history — fills 0 until warm-up complete)
        grp["turbulence"] = (close.pct_change()
                                  .rolling(252)
                                  .apply(_turbulence_1d, raw=True)
                                  .fillna(0.0))

        out_frames.append(grp)

    result = pd.concat(out_frames, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    log.info("Indicators added. Shape: %s", result.shape)
    return result


def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    tickers   = cfg["tickers"]
    start     = cfg["data"]["start_date"]
    end       = cfg["data"]["end_date"]
    out_dir   = cfg["data"]["raw_prices_dir"]
    os.makedirs(out_dir, exist_ok=True)

    prices_raw, failed = fetch_psx_prices(tickers, start, end)

    if failed:
        log.warning("Tickers with no data: %s", failed)

    prices_df = add_technical_indicators(prices_raw)
    out_path  = os.path.join(out_dir, "psx_prices.csv")
    prices_df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)
    return prices_df


if __name__ == "__main__":
    run()
