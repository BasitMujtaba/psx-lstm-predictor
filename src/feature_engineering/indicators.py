"""
src/feature_engineering/indicators.py
========================================
Computes technical indicators for PSX OHLCV data.

All functions accept a per-ticker DataFrame with columns:
    open, high, low, close, volume  (case-insensitive)
and return the same DataFrame with new indicator columns appended.

Why scale-free?
  Close prices differ wildly across tickers (OGDC ~100 PKR, LUCK ~800 PKR).
  Using raw prices would force the scaler to treat them differently.
  Every indicator here is either:
    (a) a ratio / percentage  -> dimensionless
    (b) a bounded oscillator  -> already in [0, 100] or similar

Indicators implemented
----------------------
  1. MACD          - momentum (% of price)
  2. RSI           - momentum oscillator [0, 100]
  3. CCI           - commodity channel index
  4. DMI / ADX     - trend strength + direction
  5. ATR %         - volatility relative to price
  6. Bollinger %B  - price position within bands
  7. OBV Change %  - volume-weighted momentum
  8. Turbulence    - Mahalanobis distance (market stress)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _normalise_cols(df):
    """
    Lower-case all column names and flatten yfinance MultiIndex columns.
    New yfinance returns MultiIndex like ("Close", "OGDC.KA").
    This flattens them to plain lowercase strings so the rest of the
    pipeline works without any special-casing.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def _check_cols(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")


# =============================================================================
# 1.  MACD  (normalised as % of close price)
# =============================================================================

def add_macd(df, fast=12, slow=26, signal=9):
    """
    MACD line   = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram   = MACD - Signal
    All three divided by close -> price-agnostic.
    """
    _check_cols(df, ["close"])
    c        = df["close"]
    ema_fast = c.ewm(span=fast,   adjust=False).mean()
    ema_slow = c.ewm(span=slow,   adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    hist     = macd - sig

    df = df.copy()
    df["macd_pct"]      = macd / c
    df["macd_sig_pct"]  = sig  / c
    df["macd_hist_pct"] = hist / c
    return df


# =============================================================================
# 2.  RSI  [0, 100]  ->  normalised to [0, 1]
# =============================================================================

def add_rsi(df, period=14):
    """
    Wilders RSI.
    rsi      : raw value in [0, 100]
    rsi_norm : scaled to [0, 1] to match the range of other features
    """
    _check_cols(df, ["close"])
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))

    df = df.copy()
    df["rsi"]      = rsi
    df["rsi_norm"] = rsi / 100
    return df


# =============================================================================
# 3.  CCI  (Commodity Channel Index)
# =============================================================================

def add_cci(df, period=20):
    """
    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    Clipped at +-300 then / 300 -> soft [-1, +1] range.
    """
    _check_cols(df, ["high", "low", "close"])
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))

    df = df.copy()
    df["cci"]      = cci
    df["cci_norm"] = cci.clip(-300, 300) / 300
    return df


# =============================================================================
# 4.  DMI  +  ADX  (Directional Movement Index)
# =============================================================================

def add_dmi(df, period=14):
    """
    +DM, -DM -> smoothed with Wilders EMA -> +DI, -DI in [0, 100]
    ADX       = Wilders EMA of DX

    di_plus_norm   - bullish directional strength   [0, 1]
    di_minus_norm  - bearish directional strength   [0, 1]
    adx_norm       - overall trend strength         [0, 1]
    di_diff_norm   - (+DI - -DI) / 100             [-1, +1]
    """
    _check_cols(df, ["high", "low", "close"])
    h  = df["high"]
    l  = df["low"]
    pc = df["close"].shift(1)

    tr = pd.concat([
        h - l,
        (h - pc).abs(),
        (l - pc).abs(),
    ], axis=1).max(axis=1)

    move_up   = h - h.shift(1)
    move_down = l.shift(1) - l

    dm_plus  = move_up.where((move_up > move_down) & (move_up > 0), 0.0)
    dm_minus = move_down.where((move_down > move_up) & (move_down > 0), 0.0)

    def _wilder(series, n):
        result = series.copy().astype(float)
        result.iloc[:n] = np.nan
        result.iloc[n]  = series.iloc[1:n + 1].sum()
        for i in range(n + 1, len(series)):
            result.iloc[i] = (
                result.iloc[i - 1] - (result.iloc[i - 1] / n) + series.iloc[i]
            )
        return result

    tr_smooth   = _wilder(tr,       period)
    dm_plus_sm  = _wilder(dm_plus,  period)
    dm_minus_sm = _wilder(dm_minus, period)

    di_plus  = 100 * dm_plus_sm  / tr_smooth.replace(0, np.nan)
    di_minus = 100 * dm_minus_sm / tr_smooth.replace(0, np.nan)
    dx       = (100 * (di_plus - di_minus).abs()
                / (di_plus + di_minus).replace(0, np.nan))
    adx      = _wilder(dx, period)

    df = df.copy()
    df["di_plus_norm"]  = di_plus  / 100
    df["di_minus_norm"] = di_minus / 100
    df["adx_norm"]      = adx      / 100
    df["di_diff_norm"]  = (di_plus - di_minus).clip(-100, 100) / 100
    return df


# =============================================================================
# 5.  ATR %  (Average True Range as % of close)
# =============================================================================

def add_atr(df, period=14):
    """
    ATR / close -> dimensionless volatility.
    Typical PSX values: 0.01 - 0.04  (1-4% daily range).
    """
    _check_cols(df, ["high", "low", "close"])
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False).mean()

    df = df.copy()
    df["atr_pct"] = atr / df["close"]
    return df


# =============================================================================
# 6.  Bollinger Bands  - %B and bandwidth
# =============================================================================

def add_bollinger(df, period=20, num_std=2.0):
    """
    bb_pct_b  = (close - lower) / (upper - lower)  -> 0 to 1 inside bands
    bb_width  = (upper - lower) / middle            -> dimensionless width
    """
    _check_cols(df, ["close"])
    mid   = df["close"].rolling(period).mean()
    std   = df["close"].rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    df = df.copy()
    df["bb_pct_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    return df


# =============================================================================
# 7.  OBV Change %  (On-Balance Volume momentum)
# =============================================================================

def add_obv(df, period=14):
    """
    OBV % change over N days -> scale-free regardless of float size.
    """
    _check_cols(df, ["close", "volume"])
    direction = np.sign(df["close"].diff())
    obv       = (direction * df["volume"]).fillna(0).cumsum()
    obv_pct   = obv.pct_change(period).replace([np.inf, -np.inf], np.nan)

    df = df.copy()
    df["obv_pct"] = obv_pct
    return df


# =============================================================================
# 8.  Turbulence Index  (Mahalanobis distance from historical mean)
# =============================================================================

def add_turbulence(df, lookback=252, min_periods=60):
    """
    Turbulence_t = (r_t - mu) @ cov_inv @ (r_t - mu).T

    r_t  = log-return vector [close_return, volume_return]
    mu   = rolling mean over lookback window
    cov  = rolling covariance over lookback window

    High value -> market behaving unlike its recent history
    (crash, regime change, geopolitical shock).
    Raw score returned; normalisation handled by scaler.py.
    """
    _check_cols(df, ["close", "volume"])

    log_ret_close  = np.log(df["close"] / df["close"].shift(1))
    log_ret_volume = np.log(
        df["volume"].replace(0, np.nan)
        / df["volume"].shift(1).replace(0, np.nan)
    )
    returns = pd.DataFrame({
        "ret_close":  log_ret_close,
        "ret_volume": log_ret_volume,
    }).fillna(0)

    turbulence = pd.Series(np.nan, index=df.index, name="turbulence")

    for i in range(min_periods, len(returns)):
        start = max(0, i - lookback)
        hist  = returns.iloc[start:i]
        curr  = returns.iloc[i].values
        if hist.shape[0] < min_periods:
            continue
        mu  = hist.mean().values
        cov = hist.cov().values
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue
        diff = (curr - mu).reshape(1, -1)
        turbulence.iloc[i] = max(float(diff @ cov_inv @ diff.T), 0.0)

    df = df.copy()
    df["turbulence"] = turbulence
    return df


# =============================================================================
# Master function  -  apply all indicators in one call
# =============================================================================

def add_all_indicators(df, cfg=None):
    """
    Applies every indicator to a single-ticker OHLCV DataFrame.
    Handles yfinance MultiIndex columns automatically.

    config.yaml section:
        indicators:
          macd_fast:               12
          macd_slow:               26
          macd_signal:              9
          rsi_period:              14
          cci_period:              20
          dmi_period:              14
          atr_period:              14
          bb_period:               20
          bb_std:                 2.0
          obv_period:              14
          turbulence_lookback:    252
          turbulence_min_periods:  60
    """
    p  = (cfg or {}).get("indicators", {})
    df = _normalise_cols(df)

    log.debug("Adding MACD ...")
    df = add_macd(df, fast=p.get("macd_fast", 12),
                  slow=p.get("macd_slow", 26),
                  signal=p.get("macd_signal", 9))

    log.debug("Adding RSI ...")
    df = add_rsi(df, period=p.get("rsi_period", 14))

    log.debug("Adding CCI ...")
    df = add_cci(df, period=p.get("cci_period", 20))

    log.debug("Adding DMI / ADX ...")
    df = add_dmi(df, period=p.get("dmi_period", 14))

    log.debug("Adding ATR ...")
    df = add_atr(df, period=p.get("atr_period", 14))

    log.debug("Adding Bollinger Bands ...")
    df = add_bollinger(df, period=p.get("bb_period", 20),
                       num_std=p.get("bb_std", 2.0))

    log.debug("Adding OBV ...")
    df = add_obv(df, period=p.get("obv_period", 14))

    log.debug("Adding Turbulence ...")
    df = add_turbulence(df, lookback=p.get("turbulence_lookback", 252),
                        min_periods=p.get("turbulence_min_periods", 60))

    base_cols      = {"open", "high", "low", "close", "volume"}
    indicator_cols = [c for c in df.columns if c not in base_cols]
    before         = len(df)
    df             = df.dropna(subset=indicator_cols)
    log.info("Warm-up rows dropped: %d -> %d (removed %d)",
             before, len(df), before - len(df))

    return df.reset_index(drop=True)


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    log.info("Smoke test: downloading OGDC.KA ...")
    raw = yf.download("OGDC.KA", start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw    = raw[["open", "high", "low", "close", "volume"]].dropna()
    result = add_all_indicators(raw)

    print("\n--- Last 5 rows ---")
    print(result.tail(5).to_string())

    print("\n--- Feature summary ---")
    base = {"open", "high", "low", "close", "volume"}
    for col in result.columns:
        if col not in base:
            print(f"  {col:25s}  min={result[col].min():+.4f}"
                  f"  max={result[col].max():+.4f}"
                  f"  nulls={result[col].isna().sum()}")
