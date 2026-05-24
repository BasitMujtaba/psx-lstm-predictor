"""
src/feature_engineering/merger.py
=======================================
Shifts news sentiment by 1 trading day (no look-ahead bias) and
joins it onto the price data for both decay and flags variants.

Also computes per-ticker sentiment from articles mentioning each
ticker by name. Tickers with < 30 matching articles fall back to
their category sentiment (energy or macro).

Outputs
-------
  data/processed/prices_news_joined_decay.csv
  data/processed/prices_news_joined_flags.csv
"""

import os
import subprocess
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE         = os.path.join(PROJECT_ROOT, "data")


# ── Ticker -> keyword mapping ─────────────────────────────────────────────────

TICKER_KEYWORDS = {
    "OGDC":   ["ogdc", "oil and gas development", "ogdcl"],
    "PPL":    ["ppl", "pakistan petroleum"],
    "MARI":   ["mari", "mari petroleum"],
    "PSO":    ["pso", "pakistan state oil"],
    "HBL":    ["hbl", "habib bank"],
    "MCB":    ["mcb", "muslim commercial bank"],
    "UBL":    ["ubl", "united bank"],
    "NBP":    ["nbp", "national bank"],
    "BAFL":   ["bafl", "bank alfalah"],
    "ENGRO":  ["engro"],
    "EFERT":  ["efert", "engro fertilizer"],
    "FATIMA": ["fatima", "fatima fertilizer"],
    "FFC":    ["ffc", "fauji fertilizer"],
    "LUCK":   ["lucky cement", "lucky"],
    "DGKC":   ["dgkc", "dg khan cement", "d.g. khan"],
    "MLCF":   ["mlcf", "maple leaf cement"],
    "PIOC":   ["pioc", "pioneer cement"],
    "NML":    ["nml", "nishat mills"],
    "NCL":    ["ncl", "nishat chunian"],
    "GATM":   ["gatm", "ghani", "ghani auto"],
    "TRG":    ["trg", "trg pakistan"],
    "SYS":    ["sys", "systems limited"],
    "AVN":    ["avn", "avanceon"],
    "SEARL":  ["searl", "searle"],
    "FEROZ":  ["feroz", "ferozsons"],
    "INDU":   ["indu", "indus motor"],
    "PSMC":   ["psmc", "pak suzuki"],
    "HUBC":   ["hub power", "hubco", "hubc"],
    "KAPCO":  ["kapco", "kot addu power"],
    "POL":    ["pol", "pakistan oilfields"],
}

ENERGY_TICKERS = {"OGDC", "PPL", "MARI", "PSO", "POL", "HUBC", "KAPCO"}
MIN_ARTICLES   = 30


# ── GitHub push ───────────────────────────────────────────────────────────────

def _push_to_github(decay_path, flags_path):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "pull", "--rebase", "origin", "main"],
            ["git", "-C", PROJECT_ROOT, "add", decay_path, flags_path],
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             "Update joined CSVs with per-ticker sentiment"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed joined CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


# ── Build per-ticker daily sentiment table ────────────────────────────────────

def _build_ticker_sentiment(news_path, trading_dates, tickers):
    log.info("Loading news: %s", news_path)
    news = pd.read_csv(news_path, parse_dates=["date"])
    news["title_lower"] = news["title"].str.lower()

    ticker_frames = []
    used_ticker   = []
    used_fallback = []

    for ticker in tickers:
        keywords = TICKER_KEYWORDS.get(ticker, [ticker.lower()])
        pattern  = "|".join(keywords)
        matched  = news[news["title_lower"].str.contains(pattern, na=False)]

        if len(matched) < MIN_ARTICLES:
            used_fallback.append(ticker)
            continue

        daily = (matched
                 .groupby("date")["sentiment_score"]
                 .mean()
                 .reset_index()
                 .rename(columns={"sentiment_score": "sentiment_ticker"}))

        shifted = (trading_dates
                   .merge(daily,
                          left_on  = "prev_trading_date",
                          right_on = "date",
                          how      = "left")
                   .rename(columns={"date_x": "date"})
                   .drop(columns=["date_y", "prev_trading_date"]))

        shifted["ticker"] = ticker
        ticker_frames.append(shifted[["date", "ticker", "sentiment_ticker"]].copy())
        used_ticker.append(ticker)

    log.info("Per-ticker sentiment built for %d tickers: %s", len(used_ticker), used_ticker)
    log.info("Fallback to category for %d tickers: %s", len(used_fallback), used_fallback)

    if ticker_frames:
        return pd.concat(ticker_frames, ignore_index=True), used_ticker, used_fallback
    else:
        empty = pd.DataFrame(columns=["date", "ticker", "sentiment_ticker"])
        return empty, [], list(tickers)


# ── Attach sentiment_ticker to a prices DataFrame ────────────────────────────

def _attach_ticker_sentiment(df_prices, ticker_sent, used_fallback, label):
    df = df_prices.copy()

    if ticker_sent.empty:
        df["sentiment_ticker"] = 0.0
        return df

    df = df.merge(ticker_sent, on=["date", "ticker"], how="left")

    if "sentiment_ticker" not in df.columns:
        log.warning("%s: sentiment_ticker missing after merge, defaulting to 0", label)
        df["sentiment_ticker"] = 0.0
        return df

    for ticker in used_fallback:
        mask = df["ticker"] == ticker
        if ticker in ENERGY_TICKERS:
            df.loc[mask, "sentiment_ticker"] = df.loc[mask, "sentiment_energy"]
        else:
            df.loc[mask, "sentiment_ticker"] = df.loc[mask, "sentiment_macro"]

    df["sentiment_ticker"] = df["sentiment_ticker"].fillna(0.0)

    zeros_pct = (df["sentiment_ticker"] == 0).mean() * 100
    log.info("%s sentiment_ticker mean=%.4f  zeros=%.1f%%  nulls=%d",
             label,
             df["sentiment_ticker"].mean(),
             zeros_pct,
             df["sentiment_ticker"].isna().sum())

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    prices = pd.read_csv(
        os.path.join(BASE, "processed", "psx_prices_processed.csv"),
        parse_dates=["date"])
    decay  = pd.read_csv(
        os.path.join(BASE, "processed", "news_aggregated_decay_catwise.csv"),
        parse_dates=["date"])
    flags  = pd.read_csv(
        os.path.join(BASE, "processed", "news_aggregated_flags.csv"),
        parse_dates=["date"])

    log.info("Prices : %s | %s -> %s",
             prices.shape,
             prices["date"].min().date(),
             prices["date"].max().date())

    trading_dates = (prices[["date"]]
                     .drop_duplicates()
                     .sort_values("date")
                     .reset_index(drop=True))
    trading_dates["prev_trading_date"] = trading_dates["date"].shift(1)
    log.info("Unique trading days: %d", len(trading_dates))

    decay_shifted = (trading_dates
                     .merge(decay,
                            left_on  = "prev_trading_date",
                            right_on = "date",
                            how      = "left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    flags_shifted = (trading_dates
                     .merge(flags,
                            left_on  = "prev_trading_date",
                            right_on = "date",
                            how      = "left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    decay_sentiment_cols = [
        "sentiment_corporate", "sentiment_energy",
        "sentiment_forex", "sentiment_macro", "news_count",
    ]
    flags_sentiment_cols = [
        "sentiment_corporate", "has_corporate",
        "sentiment_energy",    "has_energy",
        "sentiment_forex",     "has_forex",
        "sentiment_macro",     "has_macro",
        "news_count",
    ]
    decay_shifted[decay_sentiment_cols] = decay_shifted[decay_sentiment_cols].fillna(0.0)
    flags_shifted[flags_sentiment_cols] = flags_shifted[flags_sentiment_cols].fillna(0.0)

    tickers   = sorted(prices["ticker"].unique().tolist())
    news_path = os.path.join(BASE, "processed", "news_filtered.csv")

    ticker_sent, used_ticker, used_fallback = _build_ticker_sentiment(
        news_path, trading_dates, tickers,
    )

    prices_decay = prices.merge(decay_shifted, on="date", how="left")
    prices_flags = prices.merge(flags_shifted, on="date", how="left")

    prices_decay = _attach_ticker_sentiment(prices_decay, ticker_sent, used_fallback, "DECAY")
    prices_flags = _attach_ticker_sentiment(prices_flags, ticker_sent, used_fallback, "FLAGS")

    prices_decay = prices_decay.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_flags = prices_flags.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info("Final columns (DECAY): %s", prices_decay.columns.tolist())

    sample = prices_decay.groupby("ticker")["sentiment_ticker"].mean().round(4)
    log.info("Per-ticker sentiment means: %s", sample.to_dict())

    processed_dir = os.path.join(BASE, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    decay_path = os.path.join(processed_dir, "prices_news_joined_decay.csv")
    flags_path = os.path.join(processed_dir, "prices_news_joined_flags.csv")

    prices_decay.to_csv(decay_path, index=False)
    prices_flags.to_csv(flags_path, index=False)

    log.info("Saved decay -> %s  (%.1f MB)", decay_path, os.path.getsize(decay_path) / 1e6)
    log.info("Saved flags -> %s  (%.1f MB)", flags_path, os.path.getsize(flags_path) / 1e6)

    _push_to_github(decay_path, flags_path)

    return prices_decay, prices_flags


if __name__ == "__main__":
    run()
