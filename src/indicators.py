"""المؤشرات الفنية (كلها متجهة على السلسلة الزمنية كاملة)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pct(s: pd.Series, n: int = 1) -> pd.Series:
    return s / s.shift(n) - 1.0


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # لا خسائر خالص = RSI 100
    out = out.where(~((avg_down == 0) & (avg_up > 0)), 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_indicators(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """df: أعمدة open high low close volume بفهرس تاريخ. يرجع نسخة فيها المؤشرات."""
    out = df[["open", "high", "low", "close", "volume"]].copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)
    out["sma50_slope"] = out["sma50"] - out["sma50"].shift(10)

    out["rsi14"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["atr14"] = atr(h, l, c, 14)
    out["atr_pct"] = out["atr14"] / c * 100

    out["ret1"] = pct(c, 1) * 100
    out["ret5"] = pct(c, 5) * 100
    out["ret20"] = pct(c, 20) * 100
    out["ret60"] = pct(c, 60) * 100
    out["ret120"] = pct(c, 120) * 100

    out["vol_sma20"] = sma(v, 20)
    out["vol_ratio"] = v / out["vol_sma20"].replace(0.0, np.nan)
    out["value_sma20"] = (c * v).rolling(20, min_periods=10).mean()

    up_vol = v.where(c > c.shift(1), 0.0).rolling(20, min_periods=10).sum()
    dn_vol = v.where(c < c.shift(1), 0.0).rolling(20, min_periods=10).sum()
    ratio = up_vol / dn_vol.replace(0.0, np.nan)
    ratio = ratio.where(~((dn_vol == 0) & (up_vol > 0)), 5.0)
    out["updown_vol"] = ratio.clip(0, 5)

    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    out["obv_rising"] = (obv > obv.shift(20)).astype(float)

    out["high20"] = h.rolling(20, min_periods=15).max().shift(1)   # أعلى 20 يوم سابقة
    out["high60"] = h.rolling(60, min_periods=30).max()
    out["high252"] = h.rolling(252, min_periods=60).max()
    out["low10"] = l.rolling(10, min_periods=5).min()
    out["dist_high252"] = c / out["high252"] - 1.0
    # خصائص إضافية لطبقة التعلم الآلي (كلها سببية: لا تنظر للمستقبل)
    out["dist_sma200"] = c / out["sma200"] - 1.0
    out["breakout_dist"] = c / out["high20"] - 1.0
    return out

