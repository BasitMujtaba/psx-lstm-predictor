"""
src/data_collection/psx_downloader.py
=======================================
Fetches OHLCV price data for PSX tickers via dps.psx.com.pk,
derives LDCP / Change / Change(%) from close prices,
and computes technical indicators.

Saves:
  data/raw/psx_prices/psx_prices_raw.csv      <- OHLCV + LDCP + Change + Change(%)
  data/processed/psx_prices_processed.csv     <- raw cols + all indicators (warmup rows dropped)

Warmup rows dropped per ticker:
  - First 252 rows dropped (turbulence warmup — longest window)
  - Then any remaining NaNs dropped (rsi, cci, bb_*, dmi_dx, ldcp)

Final CSV sorted by [date, ticker] for clean train/test splits.
"""

import os, asyncio, logging, warnings, subprocess
import numpy as np
import pandas as pd
import aiohttp
import nest_asyncio
import yaml
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

_NUMERIC_COLS  = ["open", "high", "low", "close", "volume"]
_EMA_PERIODS   = [9, 21, 50, 200]
_CONCURRENCY   = 4
_TICKER_PAUSE  = 1.5
_429_WAIT      = 8.0
_RETRY_BACKOFF = 2.0
_WARMUP_ROWS   = 252


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


def _cache_valid(path):
    """
    Returns (is_valid, df_or_None).
    If cache exists and is valid, also ensures it is sorted by [date, ticker].
    If sort order was wrong, fixes and re-saves in place before returning.
    """
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return False, None
        log.info("Cache hit — CSV covers %s -> %s (%d rows)",
                 df["date"].min().date(), df["date"].max().date(), len(df))

        # ── Auto-fix sort order if needed
        df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        if not df.equals(df_sorted):
            log.info("Cache sort order was wrong — fixing to [date, ticker] and re-saving ...")
            df_sorted.to_csv(path, index=False)
            log.info("Cache re-saved with correct sort order.")
            return True, df_sorted

        return True, df
    except Exception as e:
        log.warning("Cache check failed: %s", e)
        return False, None


def _push_to_github(raw_path, processed_path, start, end):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "add", raw_path, processed_path],
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             f"Update PSX prices cache {start} -> {end}"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed updated CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


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
    df = df.copy()
    df["ldcp"]       = df.groupby("ticker")["close"].shift(1)
    df["change"]     = (df["close"] - df["ldcp"]).round(2)
    df["change_pct"] = ((df["change"] / df["ldcp"]) * 100).round(2)
    col_order = ["date", "ticker", "ldcp", "open", "high", "low", "close",
                 "change", "change_pct", "volume"]
    return df[col_order]


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
        after_nan, after_warmup - after_nan
    )
    return df


async def _fetch_month_async(session, semaphore, symbol, year, month, retries=4):
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
                        await asyncio.sleep(_429_WAIT)
                        continue
                    resp.raise_for_status()
                    return _parse_table(await resp.text(), symbol)
            except aiohttp.ClientResponseError as exc:
                log.warning("[%s %d-%02d] attempt %d/%d HTTP %s", symbol, year, month, attempt, retries, exc.status)
            except Exception as exc:
                log.warning("[%s %d-%02d] attempt %d/%d: %s", symbol, year, month, attempt, retries, exc)
            if attempt < retries:
                await asyncio.sleep(wait)
                wait *= _RETRY_BACKOFF
    return pd.DataFrame()


async def _fetch_ticker_async(session, symbol, months, semaphore):
    frames = await asyncio.gather(*[
        _fetch_month_async(session, semaphore, symbol, y, m) for y, m in months
    ])
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["date", "ticker"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def fetch_psx_prices(tickers, start, end, concurrency=_CONCURRENCY):
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    months   = list(_months_in_range(start_dt, end_dt))
    log.info("Fetching %d ticker(s) x %d month(s) | concurrency=%d | pause=%.1fs",
             len(tickers), len(months), concurrency, _TICKER_PAUSE)
    nest_asyncio.apply()

    async def _run():
        semaphore = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=concurrency * 2, ssl=False)
        valid, failed = [], []
        async with aiohttp.ClientSession(headers=_HEADERS, connector=connector) as session:
            try:
                async with session.get(PSX_PORTAL, timeout=aiohttp.ClientTimeout(total=10)):
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
                        log.info("[%s] %d rows", symbol, len(df))
                await asyncio.sleep(_TICKER_PAUSE)
        return valid, failed

    valid, failed = asyncio.run(_run())
    if not valid:
        raise RuntimeError("No data fetched for any ticker.")
    result = pd.concat(valid, ignore_index=True)

    # ── Sort by [date, ticker] first so raw CSV is date-ordered before any operations
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = _derive_ldcp_change(result)
    log.info("Raw shape: %s | tickers: %d | failed: %d",
             result.shape, result["ticker"].nunique(), len(failed))
    return result, failed


def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu, sig = np.mean(window[:-1]), np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing technical indicators for %d tickers ...", df["ticker"].nunique())

    # ── Ensure input is sorted by [date, ticker] before per-ticker processing
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    frames = []
    for ticker, grp in tqdm(df.groupby("ticker"), desc="Indicators"):
        g                = grp.copy().sort_values("date").reset_index(drop=True)
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
        atr14       = tr.rolling(14).mean()
        pdi14       = (100 * pd.Series(pos_dm, index=g.index).rolling(14).mean()
                       / atr14.replace(0, np.nan))
        ndi14       = (100 * pd.Series(neg_dm, index=g.index).rolling(14).mean()
                       / atr14.replace(0, np.nan))
        g["dmi_dx"] = (100 * (pdi14 - ndi14).abs() / (pdi14 + ndi14).replace(0, np.nan))

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

    # ── Sort by [date, ticker] before warmup drop so processed CSV is date-ordered
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    result = _drop_warmup_rows(result)

    # ── Final sort by [date, ticker] — ready for model training
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    indicator_cols = (["macd", "rsi", "cci", "dmi_dx"]
                      + [f"ema_{p}" for p in _EMA_PERIODS]
                      + ["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct", "turbulence"])
    nan_report = result[indicator_cols].isna().mean().mul(100).round(1).to_string()
    log.info("NaN per indicator (should all be 0.0 after warmup drop): %s", nan_report)
    return result


def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    tickers        = cfg["data"]["tickers"]
    start          = cfg["data"]["start_date"]
    end            = cfg["data"]["end_date"]
    raw_dir        = _resolve(cfg["data"]["raw_prices_dir"])
    processed_dir  = _resolve(cfg["data"]["processed_dir"])
    raw_path       = os.path.join(raw_dir,       "psx_prices_raw.csv")
    processed_path = os.path.join(processed_dir, "psx_prices_processed.csv")

    os.makedirs(raw_dir,       exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    is_valid, cached_df = _cache_valid(processed_path)
    if is_valid:
        log.info("Cache valid — returning processed prices (sorted by [date, ticker])")
        return cached_df

    raw_df, failed = fetch_psx_prices(tickers, start, end)
    if failed:
        log.warning("Tickers with no data: %s", failed)

    # ── Raw CSV saved already sorted by [date, ticker] from fetch_psx_prices
    raw_df.to_csv(raw_path, index=False)
    log.info("Raw saved -> %s  (%d rows, %.1f MB)",
             raw_path, len(raw_df), os.path.getsize(raw_path) / 1e6)

    processed_df = add_technical_indicators(raw_df)

    log.info("Final CSV sorted by [date, ticker] — ready for model training")
    processed_df.to_csv(processed_path, index=False)
    log.info("Processed saved -> %s  (%d rows, %.1f MB)",
             processed_path, len(processed_df), os.path.getsize(processed_path) / 1e6)

    _push_to_github(raw_path, processed_path, start, end)

    return processed_df


if __name__ == "__main__":
    run()
