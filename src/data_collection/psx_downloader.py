"""
src/data_collection/psx_downloader.py
=======================================
Fetches OHLCV price data for PSX tickers via dps.psx.com.pk,
derives LDCP / Change / Change(%) from close prices,
and computes technical indicators.

Saves:
  data/raw/psx_prices/psx_prices_raw.csv      <- OHLCV + LDCP + Change + Change(%)
  data/processed/psx_prices_processed.csv     <- raw cols + all indicators
"""

import os, asyncio, logging, warnings
import numpy as np
import pandas as pd
import aiohttp
import nest_asyncio
import yaml
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

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

# Actual PSX historicalTable columns (6 cols):
# 0:Date | 1:Open | 2:High | 3:Low | 4:Close | 5:Volume
_NUMERIC_COLS = ["open", "high", "low", "close", "volume"]

# EMA periods
_EMA_PERIODS = [9, 21, 50, 200]

# Rate-limit settings
_CONCURRENCY   = 4
_TICKER_PAUSE  = 1.5
_429_WAIT      = 8.0
_RETRY_BACKOFF = 2.0


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _parse_table(html: str, symbol: str) -> pd.DataFrame:
    """
    Parse PSX historicalTable HTML.
    API returns 6 cols: DATE | OPEN | HIGH | LOW | CLOSE | VOLUME
    LDCP / Change / Change(%) are derived separately after fetch.
    """
    soup  = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="historicalTable")

    if not table or not table.find("tbody"):
        return pd.DataFrame()

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        rows.append({
            "date":   tds[0].get_text(strip=True),
            "open":   tds[1].get_text(strip=True).replace(",", ""),
            "high":   tds[2].get_text(strip=True).replace(",", ""),
            "low":    tds[3].get_text(strip=True).replace(",", ""),
            "close":  tds[4].get_text(strip=True).replace(",", ""),
            "volume": tds[5].get_text(strip=True).replace(",", ""),
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


def _derive_ldcp_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive LDCP, Change, Change(%) per ticker from close prices.

      ldcp        = previous trading day close
      change      = close - ldcp
      change_pct  = (change / ldcp) * 100  rounded to 2 dp

    First row per ticker is NaN (no prior day) — correct behaviour.
    """
    df = df.copy()
    df["ldcp"]       = df.groupby("ticker")["close"].shift(1)
    df["change"]     = (df["close"] - df["ldcp"]).round(2)
    df["change_pct"] = ((df["change"] / df["ldcp"]) * 100).round(2)

    # Reorder columns: date, ticker, ldcp, open, high, low, close, change, change_pct, volume
    col_order = ["date", "ticker", "ldcp",
                 "open", "high", "low", "close",
                 "change", "change_pct", "volume"]
    return df[col_order]


# ── Async fetch: one month ────────────────────────────────────────────────────

async def _fetch_month_async(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol:    str,
    year:      int,
    month:     int,
    retries:   int = 4,
) -> pd.DataFrame:
    async with semaphore:
        wait = _RETRY_BACKOFF
        for attempt in range(1, retries + 1):
            try:
                async with session.post(
                    PSX_HISTORICAL,
                    json={"symbol": symbol, "month": str(month), "year": str(year)},
                    timeout=aiohttp.ClientTimeout(total=25),
                ) as resp:

                    if resp.status == 429:
                        log.debug("[%s %d-%02d] 429 — waiting %.1fs",
                                  symbol, year, month, _429_WAIT)
                        await asyncio.sleep(_429_WAIT)
                        continue

                    resp.raise_for_status()
                    html = await resp.text()
                    return _parse_table(html, symbol)

            except aiohttp.ClientResponseError as exc:
                log.warning("[%s %d-%02d] attempt %d/%d HTTP %s",
                            symbol, year, month, attempt, retries, exc.status)
            except Exception as exc:
                log.warning("[%s %d-%02d] attempt %d/%d: %s",
                            symbol, year, month, attempt, retries, exc)

            if attempt < retries:
                await asyncio.sleep(wait)
                wait *= _RETRY_BACKOFF

    return pd.DataFrame()


# ── Async fetch: one ticker (all months) ─────────────────────────────────────

async def _fetch_ticker_async(
    session:   aiohttp.ClientSession,
    symbol:    str,
    months:    list,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    tasks  = [_fetch_month_async(session, semaphore, symbol, y, m)
              for y, m in months]
    frames = await asyncio.gather(*tasks)
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["date", "ticker"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ── Step 1: Fetch raw OHLCV + derive LDCP/Change ─────────────────────────────

def fetch_psx_prices(
    tickers:     list,
    start:       str,
    end:         str,
    concurrency: int = _CONCURRENCY,
) -> tuple:
    """
    Downloads OHLCV for all tickers then derives LDCP, Change, Change(%).
    Returns (raw_df, failed_tickers).

    raw_df columns:
      date, ticker, ldcp, open, high, low, close, change, change_pct, volume
    """
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    months   = list(_months_in_range(start_dt, end_dt))

    log.info(
        "Fetching %d ticker(s) × %d month(s)  |  concurrency=%d  |  pause=%.1fs",
        len(tickers), len(months), concurrency, _TICKER_PAUSE,
    )

    nest_asyncio.apply()

    async def _run():
        semaphore = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=concurrency * 2, ssl=False)
        valid, failed = [], []

        async with aiohttp.ClientSession(
            headers=_HEADERS, connector=connector
        ) as session:
            try:
                async with session.get(PSX_PORTAL,
                                       timeout=aiohttp.ClientTimeout(total=10)):
                    pass
            except Exception:
                pass

            for ticker in tqdm(tickers, desc="Tickers"):
                symbol = _clean_ticker(ticker)
                df     = await _fetch_ticker_async(session, symbol, months, semaphore)

                if df.empty:
                    log.warning("[%s] no data — skipping", symbol)
                    failed.append(ticker)
                else:
                    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
                    df   = df.loc[mask].reset_index(drop=True)
                    if df.empty:
                        failed.append(ticker)
                    else:
                        valid.append(df)
                        log.info("[%s] ✓ %d rows", symbol, len(df))

                await asyncio.sleep(_TICKER_PAUSE)

        return valid, failed

    valid, failed = asyncio.run(_run())

    if not valid:
        raise RuntimeError("No data fetched for any ticker.")

    # ── Concat all tickers ────────────────────────────────────────────────────
    result = pd.concat(valid, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # ── Derive LDCP, Change, Change(%) and reorder columns ───────────────────
    result = _derive_ldcp_change(result)

    log.info(
        "Raw shape: %s | tickers: %d | failed: %d",
        result.shape, result["ticker"].nunique(), len(failed),
    )
    return result, failed


# ── Step 2: Technical indicators ─────────────────────────────────────────────

def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu, sig = np.mean(window[:-1]), np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends indicators to the raw DataFrame (which already contains
    ldcp / change / change_pct).

    Momentum / trend  : macd, rsi, cci, dmi_dx
    Moving averages   : ema_9, ema_21, ema_50, ema_200
    Volatility        : bb_mid, bb_upper, bb_lower, bb_width, bb_pct
    Risk              : turbulence
    """
    log.info("Computing technical indicators for %d tickers …",
             df["ticker"].nunique())
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

        # EMAs
        for period in _EMA_PERIODS:
            g[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # Bollinger Bands (20-period, 2σ)
        bb_sma        = close.rolling(20).mean()
        bb_std        = close.rolling(20).std(ddof=0)
        g["bb_mid"]   = bb_sma
        g["bb_upper"] = bb_sma + 2 * bb_std
        g["bb_lower"] = bb_sma - 2 * bb_std
        bb_range      = (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
        g["bb_width"] = bb_range / g["bb_mid"].replace(0, np.nan)
        g["bb_pct"]   = (close - g["bb_lower"]) / bb_range

        # Turbulence
        g["turbulence"] = (close.pct_change()
                               .rolling(252)
                               .apply(_turbulence_1d, raw=True)
                               .fillna(0.0))
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    indicator_cols = (
        ["macd", "rsi", "cci", "dmi_dx"]
        + [f"ema_{p}" for p in _EMA_PERIODS]
        + ["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct", "turbulence"]
    )
    nan_pct = result[indicator_cols].isna().mean().mul(100).round(1)
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

    # ── Step 1: fetch + derive LDCP/Change -> save raw ────────────────────────
    raw_df, failed = fetch_psx_prices(tickers, start, end)
    if failed:
        log.warning("Tickers with no data: %s", failed)

    raw_path = os.path.join(raw_dir, "psx_prices_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    log.info(
        "✓ Raw saved          -> %s  (%d rows, %.1f MB)  cols: %s",
        raw_path, len(raw_df),
        os.path.getsize(raw_path) / 1e6,
        list(raw_df.columns),
    )

    # ── Step 2: add indicators -> save processed ──────────────────────────────
    processed_df   = add_technical_indicators(raw_df)
    processed_path = os.path.join(processed_dir, "psx_prices_processed.csv")
    processed_df.to_csv(processed_path, index=False)
    log.info(
        "✓ Processed saved    -> %s  (%d rows, %.1f MB)  cols: %s",
        processed_path, len(processed_df),
        os.path.getsize(processed_path) / 1e6,
        list(processed_df.columns),
    )

    return processed_df


if __name__ == "__main__":
    run()
