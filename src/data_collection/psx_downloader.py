"""
src/data_collection/psx_downloader.py
=======================================
Fetches OHLCV + LDCP, Change, Change(%) price data for PSX tickers
via dps.psx.com.pk and computes technical indicators.

Saves:
  data/raw/psx_prices/psx_prices_raw.csv      <- raw OHLCV only
  data/processed/psx_prices_processed.csv     <- OHLCV + indicators
"""

import os, time, logging, warnings
import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

PSX_PORTAL     = "https://dps.psx.com.pk"
PSX_HISTORICAL = f"{PSX_PORTAL}/historical"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://dps.psx.com.pk/",
    "Origin":          "https://dps.psx.com.pk",
}

# PSX historicalTable column order:
# 0:Date | 1:LDCP | 2:Open | 3:High | 4:Low | 5:Close | 6:Change | 7:Change% | 8:Volume
_NUMERIC_COLS = ["ldcp", "open", "high", "low", "close", "change", "change_pct", "volume"]


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(PSX_PORTAL, timeout=10)
    except Exception as exc:
        log.warning("Seed request failed (non-fatal): %s", exc)
    return s


def _clean_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    for suffix in (".KA", ".PK"):
        if ticker.endswith(suffix):
            ticker = ticker[: -len(suffix)]
            break
    return ticker


def _months_in_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


# ── Per-month fetch ───────────────────────────────────────────────────────────

def _fetch_month(session, symbol, year, month, retries=3, backoff=2.0):
    wait = backoff
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                PSX_HISTORICAL,
                json={"symbol": symbol, "month": str(month), "year": str(year)},
                timeout=20,
            )
            resp.raise_for_status()

            soup  = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", id="historicalTable")

            if not table or not table.find("tbody"):
                return pd.DataFrame()

            rows = []
            for tr in table.find("tbody").find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 9:
                    continue
                rows.append({
                    "date":       tds[0].get_text(strip=True),
                    "ldcp":       tds[1].get_text(strip=True).replace(",", ""),
                    "open":       tds[2].get_text(strip=True).replace(",", ""),
                    "high":       tds[3].get_text(strip=True).replace(",", ""),
                    "low":        tds[4].get_text(strip=True).replace(",", ""),
                    "close":      tds[5].get_text(strip=True).replace(",", ""),
                    "change":     tds[6].get_text(strip=True).replace(",", ""),
                    "change_pct": tds[7].get_text(strip=True).replace(",", "").replace("%", ""),
                    "volume":     tds[8].get_text(strip=True).replace(",", ""),
                })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in _NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(subset=["date", "close"], inplace=True)
            df["ticker"] = symbol
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as exc:
            log.warning("[%s %d-%02d] attempt %d/%d: %s",
                        symbol, year, month, attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait)
                wait *= 2

    return pd.DataFrame()


# ── Step 1: Fetch raw OHLCV ───────────────────────────────────────────────────

def fetch_psx_prices(tickers, start, end, sleep=0.35):
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    months   = list(_months_in_range(start_dt, end_dt))

    log.info("Fetching %d tickers × %d months from %s",
             len(tickers), len(months), PSX_PORTAL)

    session    = _make_session()
    valid_rows = []
    failed     = []

    for ticker in tqdm(tickers, desc="Tickers"):
        symbol    = _clean_ticker(ticker)
        month_dfs = []

        for year, month in tqdm(months, desc=f"  {symbol}", leave=False):
            df = _fetch_month(session, symbol, year, month)
            if not df.empty:
                month_dfs.append(df)
            time.sleep(sleep)

        if not month_dfs:
            log.warning("[%s] No data fetched – skipping", symbol)
            failed.append(ticker)
            continue

        combined = pd.concat(month_dfs, ignore_index=True)
        combined.drop_duplicates(subset=["date", "ticker"], inplace=True)
        combined.sort_values("date", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        mask     = (combined["date"] >= start_dt) & (combined["date"] <= end_dt)
        combined = combined.loc[mask].reset_index(drop=True)

        if combined.empty:
            failed.append(ticker)
        else:
            valid_rows.append(combined)

    if not valid_rows:
        raise RuntimeError("No data fetched for any ticker.")

    result = pd.concat(valid_rows, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    log.info("Raw OHLCV shape: %s | loaded: %d | failed: %d",
             result.shape, result["ticker"].nunique(), len(failed))
    return result, failed


# ── Step 2: Technical indicators ─────────────────────────────────────────────

def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu, sig = np.mean(window[:-1]), np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_technical_indicators(df):
    log.info("Computing technical indicators")
    frames = []

    for ticker, grp in tqdm(df.groupby("ticker"), desc="Indicators"):
        g                = grp.copy().sort_values("date").reset_index(drop=True)
        close, high, low = g["close"], g["high"], g["low"]

        # MACD
        g["macd"] = (close.ewm(span=12, adjust=False).mean()
                     - close.ewm(span=26, adjust=False).mean())

        # RSI-14
        delta    = close.diff()
        rs       = (delta.clip(lower=0).rolling(14).mean()
                    / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
        g["rsi"] = 100 - (100 / (1 + rs))

        # CCI-20
        tp       = (high + low + close) / 3
        mad      = tp.rolling(20).apply(
                       lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        g["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

        # DMI / DX-14
        prev_close = close.shift(1)
        up_move    = high - high.shift(1)
        dn_move    = low.shift(1) - low
        pos_dm     = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm     = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr         = pd.concat([high - low,
                                 (high - prev_close).abs(),
                                 (low  - prev_close).abs()], axis=1).max(axis=1)
        atr14       = tr.rolling(14).mean()
        pdi14       = (100 * pd.Series(pos_dm, index=g.index).rolling(14).mean()
                       / atr14.replace(0, np.nan))
        ndi14       = (100 * pd.Series(neg_dm, index=g.index).rolling(14).mean()
                       / atr14.replace(0, np.nan))
        g["dmi_dx"] = (100 * (pdi14 - ndi14).abs()
                       / (pdi14 + ndi14).replace(0, np.nan))

        # Turbulence
        g["turbulence"] = (close.pct_change()
                               .rolling(252)
                               .apply(_turbulence_1d, raw=True)
                               .fillna(0.0))
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    nan_pct = (result[["macd", "rsi", "cci", "dmi_dx", "turbulence"]]
               .isna().mean().mul(100).round(1))
    log.info("NaN %% per indicator:\n%s", nan_pct.to_string())
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    tickers = cfg["data"]["tickers"]
    start   = cfg["data"]["start_date"]
    end     = cfg["data"]["end_date"]

    raw_dir       = _resolve(cfg["data"]["raw_prices_dir"])
    processed_dir = _resolve(cfg["data"]["processed_dir"])
    os.makedirs(raw_dir,       exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    log.info("raw_dir       -> %s", raw_dir)
    log.info("processed_dir -> %s", processed_dir)

    raw_df, failed = fetch_psx_prices(tickers, start, end)
    if failed:
        log.warning("Tickers with no data: %s", failed)

    raw_path = os.path.join(raw_dir, "psx_prices_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    log.info("✓ Raw OHLCV saved    -> %s  (%d rows, %.1f MB)",
             raw_path, len(raw_df), os.path.getsize(raw_path) / 1e6)

    processed_df   = add_technical_indicators(raw_df)
    processed_path = os.path.join(processed_dir, "psx_prices_processed.csv")
    processed_df.to_csv(processed_path, index=False)
    log.info("✓ Processed saved    -> %s  (%d rows, %.1f MB)",
             processed_path, len(processed_df),
             os.path.getsize(processed_path) / 1e6)

    return processed_df


if __name__ == "__main__":
    run()
