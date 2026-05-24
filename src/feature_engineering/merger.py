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

# tickers whose category fallback is energy sentiment
ENERGY_TICKERS = {"OGDC", "PPL", "MARI", "PSO", "POL", "HUBC", "KAPCO"}

MIN_ARTICLES = 30   # minimum articles to use per-ticker sentiment


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


# ── Build per-ticker daily sentiment ─────────────────────────────────────────

def _build_ticker_sentiment(news_path, trading_dates, tickers):
    """
    For each ticker, finds articles mentioning it by name,
    aggregates daily mean sentiment, shifts by 1 trading day,
    and returns a DataFrame with columns: date, ticker, sentiment_ticker.

    Tickers with < MIN_ARTICLES total fall back to category sentiment
    (handled later in run() during the merge).
    """
    log.info("Loading news for per-ticker sentiment: %s", news_path)
    news = pd.read_csv(news_path, parse_dates=["date"])
    news["title_lower"] = news["title"].str.lower()

    frames = []
    used_ticker_sentiment = []
    used_fallback         = []

    for ticker in tickers:
        keywords = TICKER_KEYWORDS.get(ticker, [ticker.lower()])
        pattern  = "|".join(keywords)
        mask     = news["title_lower"].str.contains(pattern, na=False)
        matched  = news[mask]

        if len(matched) < MIN_ARTICLES:
            used_fallback.append(ticker)
            continue

        # daily mean sentiment
        daily = (matched
                 .groupby("date")["sentiment_score"]
                 .mean()
                 .reset_index()
                 .rename(columns={"sentiment_score": "sentiment_ticker_raw"}))

        # shift by 1 trading day using trading calendar
        shifted = (trading_dates
                   .merge(daily,
                          left_on="prev_trading_date",
                          right_on="date",
                          how="left")
                   .rename(columns={"date_x": "date"})
                   .drop(columns=["date_y", "prev_trading_date"]))

        shifted["ticker"]            = ticker
        shifted["sentiment_ticker"]  = shifted["sentiment_ticker_raw"].fillna(np.nan)
        frames.append(shifted[["date", "ticker", "sentiment_ticker"]])
        used_ticker_sentiment.append(ticker)

    log.info("Per-ticker sentiment built for: %s", used_ticker_sentiment)
    log.info("Falling back to category sentiment for: %s", used_fallback)

    if frames:
        return pd.concat(frames, ignore_index=True), used_ticker_sentiment, used_fallback
    else:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_ticker"]), [], tickers


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # ── Load files
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

    # ── Build trading calendar
    trading_dates = (prices[["date"]]
                     .drop_duplicates()
                     .sort_values("date")
                     .reset_index(drop=True))
    trading_dates["prev_trading_date"] = trading_dates["date"].shift(1)
    log.info("Unique trading days: %d", len(trading_dates))

    # ── Shift category sentiment by 1 trading day
    decay_shifted = (trading_dates
                     .merge(decay,
                            left_on="prev_trading_date",
                            right_on="date",
                            how="left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    flags_shifted = (trading_dates
                     .merge(flags,
                            left_on="prev_trading_date",
                            right_on="date",
                            how="left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    # ── Fill NaNs for category sentiment
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

    # ── Build per-ticker sentiment
    tickers    = prices["ticker"].unique().tolist()
    news_path  = os.path.join(BASE, "processed", "news_filtered.csv")

    ticker_sent, used_ticker, used_fallback = _build_ticker_sentiment(
        news_path, trading_dates, tickers,
    )

    # ── Join category sentiment onto prices
    prices_decay = prices.merge(decay_shifted, on="date", how="left")
    prices_flags = prices.merge(flags_shifted, on="date", how="left")

    # ── Join per-ticker sentiment onto prices
    for df_prices in [prices_decay, prices_flags]:

        # merge ticker-level sentiment where available
        df_prices["sentiment_ticker"] = np.nan
        if not ticker_sent.empty:
            df_prices = df_prices.merge(
                ticker_sent,
                on=["date", "ticker"],
                how="left",
                suffixes=("", "_new"),
            )
            # use new column if it exists
            if "sentiment_ticker_new" in df_prices.columns:
                df_prices["sentiment_ticker"] = df_prices["sentiment_ticker_new"]
                df_prices.drop(columns=["sentiment_ticker_new"], inplace=True)
            else:
                df_prices["sentiment_ticker"] = df_prices["sentiment_ticker"]

        # fallback: for tickers without enough articles use category sentiment
        for ticker in used_fallback:
            mask = df_prices["ticker"] == ticker
            if ticker in ENERGY_TICKERS:
                df_prices.loc[mask, "sentiment_ticker"] =                     df_prices.loc[mask, "sentiment_energy"]
            else:
                df_prices.loc[mask, "sentiment_ticker"] =                     df_prices.loc[mask, "sentiment_macro"]

        # fill any remaining NaN with 0
        df_prices["sentiment_ticker"] = df_prices["sentiment_ticker"].fillna(0.0)

    # rebuild after in-loop reassignment
    prices_decay = prices.merge(decay_shifted, on="date", how="left")
    prices_flags = prices.merge(flags_shifted, on="date", how="left")

    # ── Attach sentiment_ticker cleanly
    for name, df_prices in [("decay", prices_decay), ("flags", prices_flags)]:
        df_prices["sentiment_ticker"] = np.nan

        if not ticker_sent.empty:
            tmp = df_prices.merge(
                ticker_sent, on=["date", "ticker"], how="left"
            )
            df_prices["sentiment_ticker"] = tmp["sentiment_ticker"].values

        for ticker in used_fallback:
            mask = df_prices["ticker"] == ticker
            if ticker in ENERGY_TICKERS:
                df_prices.loc[mask, "sentiment_ticker"] =                     df_prices.loc[mask, "sentiment_energy"]
            else:
                df_prices.loc[mask, "sentiment_ticker"] =                     df_prices.loc[mask, "sentiment_macro"]

        df_prices["sentiment_ticker"] = df_prices["sentiment_ticker"].fillna(0.0)

        log.info(
            "%s sentiment_ticker — mean=%.4f  zeros=%.1f%%  nulls=%d",
            name.upper(),
            df_prices["sentiment_ticker"].mean(),
            (df_prices["sentiment_ticker"] == 0).mean() * 100,
            df_prices["sentiment_ticker"].isna().sum(),
        )

    # ── Sort
    prices_decay = prices_decay.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_flags = prices_flags.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info("Final columns (DECAY): %s", prices_decay.columns.tolist())

    # ── Save
    processed_dir = os.path.join(BASE, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    decay_path = os.path.join(processed_dir, "prices_news_joined_decay.csv")
    flags_path = os.path.join(processed_dir, "prices_news_joined_flags.csv")

    prices_decay.to_csv(decay_path, index=False)
    prices_flags.to_csv(flags_path, index=False)

    log.info("Saved decay -> %s  (%.1f MB)",
             decay_path, os.path.getsize(decay_path) / 1e6)
    log.info("Saved flags -> %s  (%.1f MB)",
             flags_path, os.path.getsize(flags_path) / 1e6)

    _push_to_github(decay_path, flags_path)

    return prices_decay, prices_flags


if __name__ == "__main__":
    run()
