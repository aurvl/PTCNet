"""Feature engineering helpers and feature sets for OHLCV time series.

This module provides:
- `get_data_yf` : simple wrapper to download OHLCV via yfinance
- A set of small indicator helper functions (EMA, RSI Wilder, ATR, MACD helpers, etc.)
- Feature set functions `f1`, `f2`, `f3`, `f4`, `f5`, `f6` that return pandas DataFrames

The code normalizes OHLCV column names (case-insensitive) so callers may pass
DataFrames with columns like 'Open' or 'open'.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
EPS = 1e-12

# data

def get_data_yf(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV for `ticker` using yfinance.

    Returns a DataFrame indexed by datetime. Columns returned will be a subset
    of ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] depending on
    availability.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker={ticker!r} start={start!r} end={end!r} interval={interval!r}"
        )
    # Keep known columns if present and normalize the index
    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.columns = df.columns.droplevel('Ticker')
    return df


# ------------------ Helpers / indicators ------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with lowercase columns: open, high, low, close, volume, adj_close (if present).

    The function also converts the index to datetime, sorts it, and forward-fills missing values.
    """
    df = df.copy()
    # normalize column names to lowercase and strip spaces
    mapping = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    # keep canonical names if present
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI implementation (exponential smoothing).

    Returns RSI in range [0, 100].
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    au = up.ewm(alpha=1 / period, adjust=False).mean()
    ad = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = au / (ad.replace(0, np.nan) + EPS)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Compute the ordinary least squares slope on a rolling window.

    The returned slope is in the units of `y` per standardized time step.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    x = np.arange(window, dtype=float)
    x = (x - x.mean()) / (x.std() + EPS)
    yvals = y.values.astype(float)
    slopes = np.full(len(yvals), np.nan)
    for i in range(window - 1, len(yvals)):
        window_vals = yvals[i - window + 1 : i + 1]
        if not np.all(np.isfinite(window_vals)):
            continue
        slopes[i] = (x * (window_vals - window_vals.mean())).sum() / (window - 1)
    return pd.Series(slopes, index=y.index)


def _stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - ll) / (hh - ll + EPS)
    k = k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def _percentile_rank_last(window_array) -> float:
    arr = np.asarray(window_array)
    n = arr.size
    if n == 0 or not np.isfinite(arr[-1]):
        return np.nan
    return float(np.sum(arr <= arr[-1]) / n)


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, lower


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    term = np.log(high / low) ** 2
    return np.sqrt(term.rolling(window).mean() / (4 * np.log(2)))


def _rolling_skew(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.nan
    m = a.mean()
    s = a.std()
    if s <= 0:
        return 0.0
    z = (a - m) / s
    return float(np.mean(z ** 3))


def _get_data_fred(path: str, api_key: Optional[str], series_id: str, rename: str) -> pd.DataFrame:
    """Try to load a saved FRED JSON file; if not present fetch from the FRED API.

    The function saves the raw response to `path` to avoid repeated downloads.
    Returns a DataFrame indexed by date with a single column named `rename`.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame(columns=[rename])
    else:
        if not api_key:
            print('No API KEY found')
            return pd.DataFrame(columns=[rename])
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[rename])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")["value"].rename(rename).to_frame()
    df.sort_index(inplace=True)
    return df

def _compute_stop_margin_safe(
    price: float,
    Z: float,
    C: float,
    s_cap_max: float = 0.01,
    s_cap_ceiling: Optional[float] = None,
    fractional_shares: bool = True,
    min_shares: int = 1,
) -> float:
    """Compute a safe stop margin (fraction of price) for position sizing.

    Keeps result bounded and supports integer or fractional share sizing.
    """
    if price <= 0 or not np.isfinite(price):
        return float(np.nan)
    if fractional_shares:
        s_tau = (s_cap_max * C) / max(Z, EPS)
    else:
        Q = max(min_shares, int(np.floor(max(Z, EPS) / price)))
        s_tau = (s_cap_max * C) / max(Q * price, EPS)
    if s_cap_ceiling is not None:
        s_tau = min(s_tau, float(s_cap_ceiling))
    return float(np.clip(s_tau, 1e-6, 0.99))

def load_context(path) -> None:
    indic = pd.read_csv(path, index_col='Date')
    indic.index = pd.to_datetime(indic.index) 
    indic = indic.rename(columns={'Dollar': 'close'})
    df1 = indic['close']
    df2 = indic[['VIX', 'regime']].rename(columns={'VIX': 'close'})
    return df1, df2

def load_gdp_context(path: str, start: int = 2000) -> pd.DataFrame:
    """Load GDP data from World Bank CSV and transform to daily frequency.

    Args:
    ----
    path: Path to the GDP CSV file.
    start: Start year for filtering data.
    Returns:
    -------
    DataFrame with daily GDP data for selected entities.
    """
    # Vérification basique que le fichier existe
    if not os.path.exists(path):
        print(f"Warning: GDP file not found at {path}")
        return pd.DataFrame()

    gdp_df = pd.read_csv(path)
    
    # Mapping des noms longs vers des noms de features plus propres (snake_case)
    name_mapping = {
        "Arab World": "gdp_arab_world",
        "Africa Western and Central": "gdp_africa_west_central",
        "World": "gdp_world",
        "United States": "gdp_usa",
        "Europe & Central Asia": "gdp_europe_central_asia",
        "Heavily indebted poor countries: UN classification": "gdp_hipc",
        "IDA only": "gdp_ida",
        "Latin America & Caribbean": "gdp_latam_carib",
        "Least developed countries: UN classification": "gdp_ldc"
    }
    
    entities = list(name_mapping.keys())
    
    gdp_df = gdp_df[gdp_df["Country Name"].isin(entities)]
    
    # Remplacer les noms longs par les clés courtes
    gdp_df["Country Name"] = gdp_df["Country Name"].map(name_mapping)
    
    gdp_df = gdp_df.set_index("Country Name")
    
    # Sélection des années colonnes
    cols_years = [str(y) for y in range(start, pd.Timestamp.now().year + 1) if str(y) in gdp_df.columns]
    gdp_df = gdp_df[cols_years]
    
    gdp_df = gdp_df.transpose()
    gdp_df.index = pd.to_datetime(gdp_df.index, format='%Y')
    
    # Resample en fréquence Business Day et interpolation
    gdp_daily = gdp_df.resample('B').interpolate(method='linear')
    
    return gdp_daily

# ------------------ Feature sets (f1..f4) ------------------

def f1(
    df_ohlcv: pd.DataFrame,
    df_bench: Optional[Union[pd.DataFrame, pd.Series]] = None,
    df_vix: Optional[Union[pd.DataFrame, pd.Series]] = None,
    df_households: Optional[Union[pd.DataFrame, pd.Series]] = None,
    *,
    Z: float = 100.0,
    C: float = 1000.0,
    s_cap_max: float = 0.01,
    s_cap_ceiling: Optional[float] = None,
) -> pd.DataFrame:
    """Compute a core set of features from OHLCV and optional external series.

    The function normalizes input columns, computes trend/momentum/volatility
    indicators and returns a DataFrame of features indexed by date.
    """
    px = _normalize_ohlcv(df_ohlcv)
    # require core columns
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(px.columns)):
        raise ValueError(f"df_ohlcv must contain columns: {required}")

    close = px["close"].astype(float)
    high = px["high"].astype(float)
    low = px["low"].astype(float)
    open_p = px["open"].astype(float)
    volume = px["volume"].astype(float)

    # Moving averages and MACD
    sma20 = close.rolling(20, min_periods=20).mean() + EPS
    sma50 = close.rolling(50, min_periods=50).mean() + EPS
    gap_ma20 = (close - sma20) / sma20
    gap_ma50 = (close - sma50) / sma50

    slope_ma20 = _rolling_slope(sma20, 20)
    slope_ma20_norm = slope_ma20 / (sma20.rolling(20).std() + EPS)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    rsi14 = _rsi_wilder(close, 14)
    delta_rsi5 = rsi14 - rsi14.shift(5)

    stoch_k, stoch_d = _stoch_kd(high, low, close)
    stoch_spread = stoch_k - stoch_d

    # Volatility / range
    atr14 = _atr(high, low, close)
    atr14_close = atr14 / (close + EPS)

    ret1 = close.pct_change()
    log_ret_1 = np.log(close / close.shift(1))

    momentum_5d = close.pct_change(5)
    momentum_20d = close.pct_change(20)

    vol20 = ret1.rolling(20, min_periods=20).std()
    vol50 = ret1.rolling(50, min_periods=50).std()
    vol_ratio_20_50 = vol20 / (vol50 + EPS)

    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / (close + EPS)
    close_pos = ((close - low) / (high - low + EPS)).clip(0, 1)

    # Volume-based features
    vol_z20 = (volume - volume.rolling(20, min_periods=20).mean()) / (volume.rolling(20, min_periods=20).std() + EPS)
    obv = _obv(close, volume)
    obv_slope20 = _rolling_slope(obv, 20)
    abnormal_vol = (vol_z20 > 2.0).astype(int)

    overnight_gap = (open_p - close.shift(1)) / (close.shift(1) + EPS)
    skew_5d = ret1.rolling(5, min_periods=5).apply(_rolling_skew, raw=True)

    # Stop margin and risk ratios
    stop_margin = close.apply(lambda P: _compute_stop_margin_safe(P, Z, C, s_cap_max, s_cap_ceiling))
    atr14_over_stop_margin = atr14 / (stop_margin * close + EPS)

    # Relative strength vs benchmark (63 days)
    rs_63 = pd.Series(np.nan, index=close.index)
    if df_bench is not None:
        if isinstance(df_bench, pd.DataFrame):
            bench_close = _normalize_ohlcv(df_bench).get("close")
        else:
            bench_close = pd.Series(df_bench).astype(float)
        bench_close = bench_close.reindex(close.index).ffill()
        rs_63 = close.pct_change(63) - bench_close.pct_change(63)

    # VIX percentile (1 year ~ 252 trading days) and optional regime column
    vix_pct_1y = pd.Series(np.nan, index=close.index)
    market_regime = pd.Series(index=close.index, dtype="Int64")
    if df_vix is not None:
        if isinstance(df_vix, pd.DataFrame) and "close" in _normalize_ohlcv(df_vix).columns:
            vclose = _normalize_ohlcv(df_vix)["close"].reindex(close.index).ffill()
            vix_pct_1y = vclose.rolling(252, min_periods=20).apply(_percentile_rank_last, raw=True)
            vv = _normalize_ohlcv(df_vix)
            if "regime" in vv.columns:
                market_regime = vv["regime"].reindex(close.index).ffill().astype("Int64")
        else:
            vclose = pd.Series(df_vix).astype(float).reindex(close.index).ffill()
            vix_pct_1y = vclose.rolling(252, min_periods=20).apply(_percentile_rank_last, raw=True)

    # Optional households index
    if df_households is not None:
        if isinstance(df_households, pd.DataFrame):
            hh_close = _normalize_ohlcv(df_households)["close"].reindex(close.index).ffill()
        else:
            hh_close = pd.Series(df_households).astype(float).reindex(close.index).ffill()
        rel_to_households = close / (hh_close + EPS)
    else:
        rel_to_households = pd.Series(index=close.index, dtype=float)

    regime_risk = ((vix_pct_1y > 0.7) | (atr14_over_stop_margin > 0.3)).astype(int)
    regime_trend = ((sma20 > sma50) & (slope_ma20_norm > 0)).astype(int)

    features = {
        "ret1": ret1,
        "gap_ma20": gap_ma20,
        "gap_ma50": gap_ma50,
        "slope_ma20_norm": slope_ma20_norm,
        "macd_hist": macd_hist,
        "delta_rsi5": delta_rsi5,
        "stoch_spread": stoch_spread,
        "atr14_close": atr14_close,
        "vol_ratio_20_50": vol_ratio_20_50,
        "range_10": range_10,
        "close_pos": close_pos,
        "vol_z20": vol_z20,
        "obv_slope20": obv_slope20,
        "yesterday_close_logr": log_ret_1,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "overnight_gap": overnight_gap,
        "abnormal_vol": abnormal_vol,
        "skew_5d": skew_5d,
        "stop_margin": stop_margin,
        "atr14_over_stop_margin": atr14_over_stop_margin,
        "relative_strength_63": rs_63,
        "vix_percentile_1y": vix_pct_1y,
        "regime_risk": regime_risk,
        "regime_trend": regime_trend,
        "market_regime": market_regime,
        # External FRED series: load, reindex to asset dates, forward-fill and ensure numeric dtype
        # Use pd.to_numeric after ffill to avoid pandas future downcasting warning.
        "eco_pol_uncertainty": pd.to_numeric(
            _get_data_fred("data/eco_pol_uncertainty.json", os.getenv("API_KEY"), "USEPUINDXD", "eco_pol_uncertainty").reindex(close.index).ffill()["eco_pol_uncertainty"],
            errors="coerce",
        ),
        "10_years_yield": pd.to_numeric(
            _get_data_fred("data/10_years_yld.json", os.getenv("API_KEY"), "DGS10", "10_years_yield").reindex(close.index).ffill()["10_years_yield"],
            errors="coerce",
        ),
    }
    if df_households is not None:
        features["rel_to_households"] = rel_to_households

    X = pd.DataFrame(features)
    return X.dropna()


def f2(df_ohlcv: pd.DataFrame, df_vix: Optional[Union[pd.DataFrame, pd.Series]] = None, **kwargs) -> pd.DataFrame:
    """Alternative feature set (slightly different choices / scale).

    Keeps many features similar to `f1` but uses title-cased input columns as well.
    """
    px = _normalize_ohlcv(df_ohlcv)
    close = px["close"].astype(float)
    high = px["high"].astype(float)
    low = px["low"].astype(float)
    volume = px["volume"].astype(float)

    sma20 = close.rolling(20).mean() + EPS
    sma50 = close.rolling(50).mean() + EPS
    gap_ma20 = (close - sma20) / sma20
    gap_ma50 = (close - sma50) / sma50

    slope_ma20 = _rolling_slope(sma20, 20)
    slope_ma20_norm = slope_ma20 / (sma20.rolling(20).std() + EPS)

    macd_hist = (_ema(close, 12) - _ema(close, 26)) - _ema((_ema(close, 12) - _ema(close, 26)), 9)
    rsi14 = _rsi_wilder(close, 14)
    delta_rsi5 = rsi14 - rsi14.shift(5)

    stoch_k, stoch_d = _stoch_kd(high, low, close)
    stoch_spread = stoch_k - stoch_d

    atr14 = _atr(high, low, close)
    atr14_close = atr14 / (close + EPS)

    ret1 = close.pct_change()
    vol20 = ret1.rolling(20).std()
    vol50 = ret1.rolling(50).std()
    vol_ratio_20_50 = vol20 / (vol50 + EPS)

    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / (close + EPS)
    close_pos = ((close - low) / (high - low + EPS)).clip(0, 1)

    vol_z20 = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + EPS)
    obv_slope20 = _rolling_slope(_obv(close, volume), 20)

    stop_margin = close.apply(lambda P: _compute_stop_margin_safe(P, 100, 1000))
    atr_over_stop = atr14 / (stop_margin * close + EPS)

    vix_pct = pd.Series(np.nan, index=close.index)
    if df_vix is not None:
        if isinstance(df_vix, pd.DataFrame):
            vix_pct = _normalize_ohlcv(df_vix)["close"].reindex(close.index).ffill().rolling(252).apply(_percentile_rank_last, raw=True)
        else:
            vix_pct = pd.Series(df_vix).astype(float).reindex(close.index).ffill().rolling(252).apply(_percentile_rank_last, raw=True)

    X = pd.DataFrame(
        {
            "gap_ma20": gap_ma20,
            "gap_ma50": gap_ma50,
            "slope_ma20_norm": slope_ma20_norm,
            "macd_hist": macd_hist,
            "delta_rsi5": delta_rsi5,
            "stoch_spread": stoch_spread,
            "atr14_close": atr14_close,
            "vol_ratio_20_50": vol_ratio_20_50,
            "range_10": range_10,
            "close_pos": close_pos,
            "vol_z20": vol_z20,
            "obv_slope20": obv_slope20,
            "stop_margin": stop_margin,
            "atr_stop_ratio": atr_over_stop,
            "vix_pct": vix_pct,
        }
    )
    
    return X.dropna()


def f3(df_ohlcv: pd.DataFrame, df_bench: Optional[Union[pd.DataFrame, pd.Series]] = None, df_vix: Optional[Union[pd.DataFrame, pd.Series]] = None, **kwargs) -> pd.DataFrame:
    """Technical features: Bollinger, CCI, MFI, Williams %R, VWAP distance, etc."""
    px = _normalize_ohlcv(df_ohlcv)
    close = px["close"].astype(float)
    high = px["high"].astype(float)
    low = px["low"].astype(float)
    volume = px["volume"].astype(float)

    bb_up, bb_low = _bollinger_bands(close, 20, 2)
    bb_width = (bb_up - bb_low) / (close + EPS)
    bb_pct = (close - bb_low) / (bb_up - bb_low + EPS)

    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + EPS)

    raw_money_flow = tp * volume
    pos_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
    neg_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)
    pos_mf = pd.Series(pos_flow, index=px.index).rolling(14).sum()
    neg_mf = pd.Series(neg_flow, index=px.index).rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + EPS)))

    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    williams_r = -100 * (hh14 - close) / (hh14 - ll14 + EPS)

    vwap_5d = (close * volume).rolling(5).sum() / (volume.rolling(5).sum() + EPS)
    dist_vwap = (close - vwap_5d) / (vwap_5d + EPS)

    X = pd.DataFrame(
        {
            "bb_width": bb_width,
            "bb_pct": bb_pct,
            "cci": cci,
            "mfi": mfi,
            "williams_r": williams_r,
            "dist_vwap": dist_vwap,
            "close_log_ret": np.log(close / close.shift(1)),
        }
    )
    return X.dropna()


def f4(df_ohlcv: pd.DataFrame, df_bench: Optional[Union[pd.DataFrame, pd.Series]] = None, df_vix: Optional[Union[pd.DataFrame, pd.Series]] = None, **kwargs) -> pd.DataFrame:
    """Statistical / risk features: rolling moments, Parkinson vol, downside deviation, efficiency ratio."""
    px = _normalize_ohlcv(df_ohlcv)
    close = px["close"].astype(float)

    log_ret = np.log(close / close.shift(1))
    roll_skew = log_ret.rolling(60).skew()
    roll_kurt = log_ret.rolling(60).kurt()
    parkinson = _parkinson_vol(px["high"].astype(float), px["low"].astype(float), 20)

    downside = log_ret.copy()
    downside[downside > 0] = 0
    downside_dev = downside.rolling(20).std()

    change = (close - close.shift(10)).abs()
    volatility = (close - close.shift(1)).abs().rolling(10).sum()
    efficiency_ratio = change / (volatility + EPS)

    vol_z = (px["volume"] - px["volume"].rolling(20).mean()) / (px["volume"].rolling(20).std() + EPS)

    X = pd.DataFrame(
        {
            "roll_skew": roll_skew,
            "roll_kurt": roll_kurt,
            "parkinson_vol": parkinson,
            "downside_dev": downside_dev,
            "efficiency_ratio": efficiency_ratio,
            "vol_z_score": vol_z,
            "log_ret": log_ret,
        }
    )
    return X.dropna()


def f5(
    df_ohlcv: pd.DataFrame,
    df_vix: Optional[Union[pd.DataFrame, pd.Series]] = None,
    df_gdp: Optional[pd.DataFrame] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Macro & Financial features only. 
    
    Replaces technical relative strength with Global Macro Context (GDP Growth).
    
    Includes:
    - VIX percentile (1 year)
    - Market Regime (from external source)
    - Economic Policy Uncertainty (FRED)
    - 10-Year Treasury Yield (FRED)
    - CPI (FRED)
    - Global GDP Growth Rates (Year-over-Year) from df_gdp
    """
    px = _normalize_ohlcv(df_ohlcv)
    close = px["close"].astype(float)
    
    # 1. VIX Percentile (1 year ~ 252 trading days) & Market Regime
    vix_pct_1y = pd.Series(np.nan, index=close.index)
    market_regime = pd.Series(index=close.index, dtype="Int64")
    
    if df_vix is not None:
        # Handle VIX passed as DataFrame
        if isinstance(df_vix, pd.DataFrame) and "close" in _normalize_ohlcv(df_vix).columns:
            vclose = _normalize_ohlcv(df_vix)["close"].reindex(close.index).ffill()
            vix_pct_1y = vclose.rolling(252, min_periods=20).apply(_percentile_rank_last, raw=True)
            
            vv = _normalize_ohlcv(df_vix)
            if "regime" in vv.columns:
                market_regime = vv["regime"].reindex(close.index).ffill().astype("Int64")
        # Handle VIX passed as Series
        else:
            vclose = pd.Series(df_vix).astype(float).reindex(close.index).ffill()
            vix_pct_1y = vclose.rolling(252, min_periods=20).apply(_percentile_rank_last, raw=True)

    # 2. FRED Data (Macro)
    # Economic Policy Uncertainty
    eco_pol = _get_data_fred(
        "data/eco_pol_uncertainty.json", 
        os.getenv("API_KEY"), 
        "USEPUINDXD", 
        "eco_pol_uncertainty"
    ).reindex(close.index).ffill()["eco_pol_uncertainty"]

    # 10-Year Treasury Yield
    us10y = _get_data_fred(
        "data/10_years_yld.json", 
        os.getenv("API_KEY"), 
        "DGS10", 
        "10_years_yield"
    ).reindex(close.index).ffill()["10_years_yield"]

    # CPI (Consumer Price Index)
    cpi_df = _get_data_fred(
        "data/cpi_us.json", 
        os.getenv("API_KEY"), 
        "CORESTICKM159SFRBATL", 
        "cpi_us"
    )
    
    # Convert monthly CPI to daily frequency using forward fill (Step function)
    # This avoids look-ahead bias.
    cpi = cpi_df.reindex(close.index).ffill()["cpi_us"]
    
    # 3. GDP Data Integration
    # Initialize a dictionary to store GDP columns
    gdp_features = {}
    
    if df_gdp is not None and not df_gdp.empty:
        # Align GDP dates to the asset dates
        # ffill() propagates the last known value forward
        gdp_aligned = df_gdp.reindex(close.index).ffill()
        
        # GDP Feature Engineering:
        # Absolute levels (e.g., 20 Trillions) are bad for ML (non-stationary).
        # We compute the Year-over-Year growth rate => pct_change(252)
        # This captures the "pace" of the economy.
        gdp_growth = gdp_aligned.pct_change(252)
        
        for col in gdp_growth.columns:
            # Add _yoy suffix for "Year over Year"
            col_name = f"{col}_yoy" 
            gdp_features[col_name] = gdp_growth[col]

    # Final Assembly
    # Create the base dictionary with VIX, FRED data, etc.
    data_dict = {
        "vix_percentile_1y": vix_pct_1y,
        "market_regime": market_regime,
        "eco_pol_uncertainty": pd.to_numeric(eco_pol, errors="coerce"),
        "10_years_yield": pd.to_numeric(us10y, errors="coerce"),
        "cpi_us": pd.to_numeric(cpi, errors="coerce"),
    }
    
    # Add the GDP features to the dictionary (Merge)
    # This ADDS the GDP keys to the existing ones, it does not delete anything.
    data_dict.update(gdp_features)
    
    X = pd.DataFrame(data_dict)
    
    # Debugging: check for NaNs (usually at the start of the series due to rolling/pct_change)
    # print(X.isna().sum())
    
    return X.dropna()


def f6(df_ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Price Action & Calendar Features specialist.
    
    Features:
    - 5 Price Action: Log Return, Shadow Up, Shadow Down, Body Size, Gap.
    - 15+ Calendar: Month, Day, Quarter, and their Cyclic Encodings (Sin/Cos).
    """
    px = _normalize_ohlcv(df_ohlcv)
    close = px["close"].astype(float)
    open_p = px["open"].astype(float)
    high = px["high"].astype(float)
    low = px["low"].astype(float)
    
    # --- 1. PRICE ACTION (5 Features) ---
    
    # A. Rendement Log
    log_ret = np.log(close / close.shift(1))
    
    # B. Taille du corps (Body) normalisée par le prix
    # (Positif = Bougie verte, Négatif = Rouge)
    body_size = (close - open_p) / (open_p + EPS)
    
    # C. Gap à l'ouverture (Open - Close veille)
    gap = (open_p - close.shift(1)) / (close.shift(1) + EPS)
    
    # D. Ombres (Wicks) : Pression vendeuse (Haut) et acheteuse (Bas)
    # On normalise par la taille totale de la bougie (High - Low)
    candle_range = high - low + EPS
    shadow_upper = (high - np.maximum(close, open_p)) / candle_range
    shadow_lower = (np.minimum(close, open_p) - low) / candle_range

    # --- 2. CALENDAR FEATURES (Time Embeddings) ---
    # Il est CRUCIAL d'encoder le temps de manière cyclique pour un CNN/MLP.
    # Sinon le modèle pense que Mois 12 (Déc) est très loin de Mois 1 (Janv), alors qu'ils sont voisins.
    
    dates = px.index
    
    # Extraction basique
    month = dates.month
    day_of_month = dates.day
    day_of_week = dates.dayofweek # 0=Lundi, 6=Dimanche
    day_of_year = dates.dayofyear
    quarter = dates.quarter
    
    # Encodage Cyclique (Sin / Cos)
    # Formule : sin(2 * pi * val / max_val)
    
    # Mois (Cycle de 12)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)
    
    # Jour de la semaine (Cycle de 7)
    sin_day_week = np.sin(2 * np.pi * day_of_week / 7)
    cos_day_week = np.cos(2 * np.pi * day_of_week / 7)
    
    # Jour du mois (Cycle de 31)
    sin_day_month = np.sin(2 * np.pi * day_of_month / 31)
    cos_day_month = np.cos(2 * np.pi * day_of_month / 31)
    
    # Jour de l'année (Cycle de 365) -> Capture la saisonnalité large
    sin_day_year = np.sin(2 * np.pi * day_of_year / 365)
    cos_day_year = np.cos(2 * np.pi * day_of_year / 365)
    
    # Booleans (Effets de fin de période)
    is_month_end = dates.is_month_end.astype(int)
    is_quarter_end = dates.is_quarter_end.astype(int)
    is_year_start = dates.is_year_start.astype(int)
    
    # Assemblage
    X = pd.DataFrame({
        # Price Action
        "log_ret": log_ret,
        "body_rel_size": body_size,
        "gap_open": gap,
        "shadow_upper": shadow_upper,
        "shadow_lower": shadow_lower,
        
        # Calendar Raw
        "month_raw": month,
        "day_week_raw": day_of_week,
        "quarter": quarter,
        
        # Calendar Cyclic (Le plus important pour le modèle)
        "sin_month": sin_month,
        "cos_month": cos_month,
        "sin_day_week": sin_day_week,
        "cos_day_week": cos_day_week,
        "sin_day_month": sin_day_month,
        "cos_day_month": cos_day_month,
        "sin_day_year": sin_day_year,
        "cos_day_year": cos_day_year,
        
        # Calendar Events
        "is_month_end": is_month_end,
        "is_quarter_end": is_quarter_end,
        "is_year_start": is_year_start
    })
    
    return X.dropna()


# target var

def compute_position_and_sl(
    P_entry: float,
    Z: float,
    C: float,
    s_cap_max: float = 0.01,
    s_cap_ceiling: float | None = None,
) -> tuple[int, float, float]:
    """
    This function computes the position size (Q) and stop-loss level (P_SL) based on money management principles.
      - Q = floor(Z / P_entry)
      - s_tau such that Q * P_entry * s_tau <= s_cap_max * C

    Returns:
      Q      : integer quantity of shares to buy
      s_tau  : stop in % of price
      P_SL   : stop level in price
    """
    if P_entry <= 0:
        return 0, 0.0, 0.0

    Q = int(np.floor(Z / P_entry))
    if Q <= 0:
        return 0, 0.0, 0.0

    s_tau = (s_cap_max * C) / (Q * P_entry)
    if s_cap_ceiling is not None:
        s_tau = min(s_tau, float(s_cap_ceiling))

    s_tau = float(np.clip(s_tau, 1e-6, 0.99))
    P_SL = P_entry * (1.0 - s_tau)
    return Q, s_tau, P_SL

def compute_dynamic_k_series(
    df: pd.DataFrame,
    T_H: int = 63,          # horizon (3 months)
    window: int = 252,      # window for local vol (1 year)
    alpha: float = 1.0,     # vol multiplier (1x, 0.8x, 1.2x, ...)
) -> pd.Series:
    """
    Dynamically compute k_t based on local volatility estimation.

    Steps:
      1) r_t = log(Close_t / Close_{t-1})
      2) sigma_1d(t) = std( r_{t-window+1 .. t} )
      3) sigma_H(t) ≈ sigma_1d(t) * sqrt(T_H)
      4) k_t = alpha * (exp(sigma_H(t)) - 1)

    Returns a Series aligned on the index of df (with NaN at the beginning).
    """
    if "close" not in df.columns and "Close" not in df.columns:
        raise ValueError("df must contain a 'close' or 'Close' column")
    
    close = df["close"] if "close" in df.columns else df["Close"]
    close = close.astype(float).sort_index()

    r = np.log(close / close.shift(1))
    sigma_1d = r.rolling(window=window, min_periods=window).std(ddof=1)
    sigma_H = sigma_1d * np.sqrt(T_H)
    k_t = alpha * (np.exp(sigma_H) - 1.0)

    return k_t

def create_y(
    df_price: pd.DataFrame,
    T_H: int = 63,          # ~3 months
    k: float = 0.10,        # +10 % of return target
    max_dd: float = 0.15,   # 15 % of max drawdown accepted (if use_mm=False)
    *,
    use_mm: bool = False,   # True = money management capital-based
    Z: float = 100.0,
    C: float = 10000.0,
    s_cap_max: float = 0.01,
    s_cap_ceiling: float | None = None,
    return_details: bool = False,
    debug: bool = False,
):
    """
    Create a binary target y_t for each day t:

      y_t = 1 if:
        - the stop is never hit between t and t+T_H
        - and (Close_{t+T_H} - Close_t)/Close_t >= k

      y_t = 0 if the position is open and at least one condition fails.

      y_t = NaN if:
        - we are too close to the end (not enough future data)
        - and (optional) use_mm=True and Q<=0 (position not open)

    If use_mm=False:
      - the stop is based on a fixed drawdown 'max_dd' (in proportion).
    If use_mm=True:
      - the stop is calculated by compute_position_and_sl (capital-aware).

    return_details=True also returns two Series:
      - ret_T : final return R_t
      - dd_T  : max drawdown DD_t
    """
    # df_price must be a DataFrame with at least 'low' and 'close' columns
    if not hasattr(df_price, "columns"):
        raise ValueError(
            "create_y expects a DataFrame with columns ['low', 'close']. "
            "You passed a pandas Series. Convert your data to a DataFrame, e.g. "
            "pd.DataFrame({'close': close_series, 'low': low_series})"
        )

    # Detect missing columns (case-insensitive)
    colmap = {c.lower(): c for c in df_price.columns}
    required = ["low", "close"]
    missing = [c for c in required if c not in colmap]
    if missing:
        raise ValueError(f"Missing columns in df_price: {missing}")

    close = df_price[colmap["close"]].astype(float)
    low = df_price[colmap["low"]].astype(float)

    n = len(df_price)
    index = df_price.index

    y     = pd.Series(np.nan, index=index, dtype="float64")
    ret_T = pd.Series(np.nan, index=index, dtype="float64")
    dd_T  = pd.Series(np.nan, index=index, dtype="float64")

    if n <= T_H:
        return (y, ret_T, dd_T) if return_details else y

    for i in range(n - T_H):
        idx_tau = index[i]
        P_tau = float(close.iloc[i])

        # 1) stop in % (s_tau) and stop level (P_SL)
        if use_mm:
            Q, s_tau, P_SL = compute_position_and_sl(
                P_entry=P_tau,
                Z=Z,
                C=C,
                s_cap_max=s_cap_max,
                s_cap_ceiling=s_cap_ceiling,
            )
            # dead zone: no position opened
            if Q <= 0:
                if debug:
                    print(f"[DEBUG] Q<=0 at {idx_tau}, P={P_tau:.2f}")
                continue  # y, ret_T, dd_T remain NaN
        else:
            # fixed stop based on max_dd (if max_dd is None => no stop)
            if max_dd is None:
                s_tau = None
                P_SL = None
            else:
                s_tau = float(max_dd)
                P_SL = P_tau * (1.0 - s_tau)

        # 2) trajectory over [t, t+T_H]
        w_low   = low.iloc[i : i + T_H + 1]
        w_close = close.iloc[i : i + T_H + 1]

        P_min = float(np.min(w_low.values))
        P_end = float(w_close.iloc[-1])

        # max drawdown and return
        dd = float((P_min - P_tau) / (P_tau + EPS))   # <= 0
        rT = float((P_end - P_tau) / (P_tau + EPS))

        dd_T.iloc[i]  = dd
        ret_T.iloc[i] = rT

        # 3) conditions
        if s_tau is None:
            cond_no_stop = True          # no stop -> ignore DD
        else:
            cond_no_stop = bool(P_min > P_SL)

        if isinstance(k, (pd.Series, pd.DataFrame)):
            k_i = float(k.iloc[i])
        else:
            k_i = float(k)
        cond_ret = bool(rT >= k_i)

        if debug and (not np.isfinite(dd) or not np.isfinite(rT)):
            print(f"[DEBUG] Non finite at {idx_tau}: dd={dd}, rT={rT}")

        y.iloc[i] = 1.0 if (cond_no_stop and cond_ret) else 0.0

    return (y, ret_T, dd_T) if return_details else y

__all__ = ["get_data_yf", "f1", "f2", "f3", "f4", "f5", "f6", "create_y"]
