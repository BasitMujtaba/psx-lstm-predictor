"""
src/feature_engineering/indicators.py
========================================
Computes scale-free technical indicators for PSX LSTM pipeline.

Input
-----
  Per-ticker DataFrame from psx_downloader with columns:
    date, ticker, ldcp, open, high, low, close,
    change, change_pct, volume

Indicators produced
-------------------
  MACD        macd_pct, macd_sig_pct, macd_hist_pct     (% of close)
  RSI         rsi_norm                                   [0, 1]
  CCI         cci_norm                                   [-1, +1]
  DMI / ADX   di_plus_norm, di_minus_norm,
              adx_norm, di_diff_norm                     [0,1] / [-1,+1]
  ATR %       atr_pct                                    (% of close)
  Bollinger   bb_pct_b, bb_width                         [0,1] / ratio
  OBV         obv_pct                                    (% change)
  Turbulence  turbulence                                 (Mahalanobis)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Guards ────────────────────────────────────────────────────────────────────

def _check_cols(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")


# ── 1. MACD (normalised as % of close) ───────────────────────────────────────

def add_macd(df, fast=12, slow=26, signal=9):
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


# ── 2. RSI -> normalised to [0, 1] ───────────────────────────────────────────

def add_rsi(df, period=14):
    _check_cols(df, ["close"])
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))

    df = df.copy()
    df["rsi_norm"] = rsi / 100
    return df


# ── 3. CCI -> clipped [-1, +1] ───────────────────────────────────────────────

def add_cci(df, period=20):
    _check_cols(df, ["high", "low", "close"])
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))

    df = df.copy()
    df["cci_norm"] = cci.clip(-300, 300) / 300
    return df


# ── 4. DMI / ADX ─────────────────────────────────────────────────────────────

def add_dmi(df, period=14):
    """
    di_plus_norm   [0, 1]   bullish directional strength
    di_minus_norm  [0, 1]   bearish directional strength
    adx_norm       [0, 1]   overall trend strength  (divided by 100)
    di_diff_norm   [-1, +1] (+DI - -DI) / 100
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

    dm_plus  = move_up.where((move_up > move_down)   & (move_up > 0),   0.0)
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

    tr_s   = _wilder(tr,       period)
    dmp_s  = _wilder(dm_plus,  period)
    dmm_s  = _wilder(dm_minus, period)

    di_plus  = 100 * dmp_s  / tr_s.replace(0, np.nan)
    di_minus = 100 * dmm_s  / tr_s.replace(0, np.nan)
    dx       = (100 * (di_plus - di_minus).abs()
                / (di_plus + di_minus).replace(0, np.nan))
    adx      = _wilder(dx, period)

    df = df.copy()
    df["di_plus_norm"]  = di_plus  / 100
    df["di_minus_norm"] = di_minus / 100
    df["adx_norm"]      = adx      / 100   # fixed: was missing /100, raw ADX leaked into features
    df["di_diff_norm"]  = (di_plus - di_minus).clip(-100, 100) / 100
    return df


# ── 5. ATR % ─────────────────────────────────────────────────────────────────

def add_atr(df, period=14):
    _check_cols(df, ["high", "low", "close"])
    pc  = df["close"].shift(1)
    tr  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False).mean()

    df = df.copy()
    df["atr_pct"] = atr / df["close"]
    return df


# ── 6. Bollinger Bands ────────────────────────────────────────────────────────

def add_bollinger(df, period=20, num_std=2.0):
    _check_cols(df, ["close"])
    mid   = df["close"].rolling(period).mean()
    std   = df["close"].rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    df = df.copy()
    df["bb_pct_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    return df


# ── 7. OBV % change ───────────────────────────────────────────────────────────

def add_obv(df, period=14):
    _check_cols(df, ["close", "volume"])
    direction = np.sign(df["close"].diff())
    obv       = (direction * df["volume"]).fillna(0).cumsum()
    obv_pct   = obv.pct_change(period).replace([np.inf, -np.inf], np.nan)

    df = df.copy()
    df["obv_pct"] = obv_pct
    return df


# ── 8. Turbulence (Mahalanobis distance) ─────────────────────────────────────

def add_turbulence(df, lookback=252, min_periods=60):
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


# ── Master function ───────────────────────────────────────────────────────────

def add_all_indicators(df, cfg=None):
    """
    Applies all scale-free indicators to a single-ticker DataFrame.

    Input columns required:
        date, open, high, low, close, volume
        (ldcp, change, change_pct, ticker also accepted and passed through)

    config.yaml section (all optional — defaults shown):
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
    p = (cfg or {}).get("indicators", {})

    df = add_macd(df,
                  fast=p.get("macd_fast", 12),
                  slow=p.get("macd_slow", 26),
                  signal=p.get("macd_signal", 9))
    df = add_rsi(df,       period=p.get("rsi_period", 14))
    df = add_cci(df,       period=p.get("cci_period", 20))
    df = add_dmi(df,       period=p.get("dmi_period", 14))
    df = add_atr(df,       period=p.get("atr_period", 14))
    df = add_bollinger(df,
                       period=p.get("bb_period", 20),
                       num_std=p.get("bb_std", 2.0))
    df = add_obv(df,       period=p.get("obv_period", 14))
    df = add_turbulence(df,
                        lookback=p.get("turbulence_lookback", 252),
                        min_periods=p.get("turbulence_min_periods", 60))

    indicator_cols = [
        "macd_pct", "macd_sig_pct", "macd_hist_pct",
        "rsi_norm", "cci_norm",
        "di_plus_norm", "di_minus_norm", "adx_norm", "di_diff_norm",
        "atr_pct", "bb_pct_b", "bb_width", "obv_pct", "turbulence",
    ]
    before = len(df)
    df     = df.dropna(subset=indicator_cols)
    log.info("Warm-up rows dropped: %d -> %d (removed %d)",
             before, len(df), before - len(df))

    return df.reset_index(drop=True)
