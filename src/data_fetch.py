"""سحب الأسعار: Yahoo أساسي، EODHD احتياطي بميزانية طلبات، والكاش في data/prices."""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import PRICES

COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------- الكاش
def cache_path(code: str):
    return PRICES / f"{code}.csv"


def read_cache(code: str) -> pd.DataFrame | None:
    p = cache_path(code)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    # Sanitize older cached files too; don't let malformed candles poison indicators.
    clean = _clean(df)
    if len(clean) != len(df):
        print(f"  [cache] {code}: تم استبعاد {len(df) - len(clean)} شمعة غير منطقية")
    return clean.sort_index()


def write_cache(code: str, df: pd.DataFrame) -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    out = _clean(df)
    out.index.name = "date"
    out.round(4).to_csv(cache_path(code))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close", "high", "low", "open"])
    df["volume"] = df["volume"].fillna(0)
    # شموع غير منطقية: High يجب أن يغطي Open/Close وLow يجب أن يغطيهما من الأسفل.
    # هذا يكشف انزياح الأعمدة أو اختلاف تعديل الأسعار بين المصادر.
    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["volume"] >= 0)
    )
    invalid_count = int((~valid).sum())
    if invalid_count:
        print(f"  [data] تم استبعاد {invalid_count} شمعة OHLCV غير منطقية")
    return df.loc[valid]


# ---------------------------------------------------------------- Yahoo
def fetch_yahoo_from(symbol: str, start: date | str, retries: int = 3) -> pd.DataFrame | None:
    """Fetch adjusted Yahoo bars from a specific date; useful for low-cost incremental refresh."""
    import yfinance as yf  # استيراد متأخر عشان الاختبار بدون إنترنت

    start_text = start.isoformat() if isinstance(start, date) else str(start)
    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.Ticker(symbol).history(start=start_text, auto_adjust=True, actions=False)
            if raw is not None and len(raw) > 0:
                raw = raw.rename(columns=str.lower)
                return _clean(raw)
            return None
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [Yahoo] فشل {symbol}: {last_err}")
    return None


def fetch_yahoo(symbol: str, years: int, retries: int = 3) -> pd.DataFrame | None:
    start = date.today() - timedelta(days=int(years * 365.25))
    return fetch_yahoo_from(symbol, start, retries=retries)


# ---------------------------------------------------------------- EODHD
def fetch_eodhd(code: str, token: str, start: date) -> pd.DataFrame | None:
    try:
        r = requests.get(
            f"https://eodhd.com/api/eod/{code}.EGX",
            params={"api_token": token, "fmt": "json", "from": start.isoformat()},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  [EODHD] {code}: HTTP {r.status_code}")
            return None
        rows = r.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        # تعديل الأسعار بمعامل التعديل عشان تتسق مع Yahoo (auto_adjust)
        factor = (df["adjusted_close"] / df["close"]).where(df["close"] > 0, 1.0)
        for c in ("open", "high", "low"):
            df[c] = df[c] * factor
        df["close"] = df["adjusted_close"]
        return _clean(df)
    except Exception as e:  # noqa: BLE001
        print(f"  [EODHD] فشل {code}: {e}")
        return None


def _eodhd_budget(state: dict, cfg: dict) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    e = state.setdefault("eodhd", {})
    if e.get("date") != today:
        e["date"] = today
        e["calls"] = 0
    return max(0, cfg["data"]["eodhd_daily_budget"] - e["calls"])


# ---------------------------------------------------------------- OANOR (بيانات لحظية للسوق المصري)
def oanor_key() -> str:
    return (os.environ.get("OANOR_API_KEY") or os.environ.get("OANOR_API_TOKEN") or os.environ.get("OANOR_KEY") or "").strip().strip('"').strip("'")

def fetch_oanor_market() -> dict | None:
    """لقطة لحظية للسوق من OANOR؛ نجرب المفتاح ثم الاختبار العام عند 401/403."""
    key = oanor_key()
    headers = {"x-oanor-key": key} if key else {}
    base = "https://api.oanor.com/egx-api/v1"
    out = {}
    try:
        r = requests.get(f"{base}/index", headers=headers, timeout=20)
        if r.status_code in (401, 402, 403) and headers:
            r = requests.get(f"{base}/index", timeout=20)
        if r.status_code == 200:
            out["index"] = r.json()
        else:
            print(f"  [OANOR] index HTTP {r.status_code}")
        time.sleep(1.1)
        r = requests.get(f"{base}/screener", params={"sort": "change", "order": "desc", "limit": 20}, headers=headers, timeout=20)
        if r.status_code in (401, 402, 403) and headers:
            r = requests.get(f"{base}/screener", params={"sort": "change", "order": "desc", "limit": 20}, timeout=20)
        if r.status_code == 200:
            out["screener"] = r.json()
        else:
            print(f"  [OANOR] screener HTTP {r.status_code}")
        if out:
            out["source"] = "OANOR"
            return out
        return fetch_egidegypt_market()
    except Exception as e:
        print(f"  [OANOR] تعذر جلب لقطة السوق: {e}")
        return fetch_egidegypt_market()

def fetch_egidegypt_market() -> dict | None:
    """Fallback من Feed البورصة المصرية/EGI عند تعذر OANOR."""
    base = "https://ticker.egidegypt.com"
    endpoints = {
        "indices": "/api/Feed/GetEGXINDICESDETAILS",
        "summary": "/api/Feed/GetMarketSummary",
        "market_watch": "/api/Feed/GetEGXMarketWatch",
        "statistics": "/api/Feed/GetEGXStatistics",
    }
    out = {"source": "EGI Feed"}
    for name, path in endpoints.items():
        try:
            r = requests.get(base + path, timeout=20)
            if r.status_code == 200:
                out[name] = r.json()
        except Exception as e:
            print(f"  [EGI] {name}: {e}")
    return out if len(out) > 1 else None

def fetch_eodhd_index(symbol: str, token: str, years: int) -> pd.DataFrame | None:
    """EODHD index symbols already contain .INDX; لا نضيف .EGX إليها."""
    try:
        r = requests.get(
            f"https://eodhd.com/api/eod/{symbol}",
            params={"api_token": token, "fmt": "json",
                    "from": (date.today() - timedelta(days=int(years * 365.25))).isoformat()},
            timeout=30,
        )
        if r.status_code != 200:
            return None
        rows = r.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return _clean(df)
    except Exception as e:
        print(f"  [EODHD] فشل المؤشر {symbol}: {e}")
        return None

def fetch_market_indices(cfg: dict, state: dict, offline: bool = False) -> dict:
    """يجرب المؤشرات الرئيسية، ويحفظ ما ينجح منها بدون اختلاق بيانات."""
    names = {
        "EGX30": "CASE30.INDX",
        # EGX33 Shariah index history starts at its base date (2022-01-01).
        # Use it as the most relevant benchmark for the personal Sharia universe.
        "EGX33_SHARIAH": "SHARIAH.INDX",
    }
    token = _eodhd_token()
    result = {}
    for name, symbol in names.items():
        p = PRICES / f"_INDEX_{name.replace(' ', '_')}.csv"
        df = None
        if not offline and token and _eodhd_budget(state, cfg) > 0:
            state["eodhd"]["calls"] += 1
            df = fetch_eodhd_index(symbol, token, cfg["data"]["history_years"])
            if df is not None and len(df) >= 20:
                write_cache(f"_INDEX_{name.replace(' ', '_')}", df)
        if df is None:
            try:
                df = read_cache(f"_INDEX_{name.replace(' ', '_')}")
            except Exception:
                df = None
        if df is not None and len(df):
            last = df.iloc[-1]
            prev = df.iloc[-2]["close"] if len(df) > 1 else None
            result[name] = {
                "symbol": symbol, "close": float(last["close"]),
                "change_pct": float((last["close"] / prev - 1) * 100) if prev else None,
                "date": str(df.index[-1].date()),
            }
    return result

# ---------------------------------------------------------------- جلسة السوق المتوقعة

def expected_latest_session(now: datetime, cfg: dict) -> date:
    """آخر جلسة EGX يُفترض توافر إغلاقها عند وقت التشغيل."""
    market = cfg.get("market", {})
    tz = ZoneInfo(market.get("timezone", "Africa/Cairo"))
    local_now = now.replace(tzinfo=tz) if now.tzinfo is None else now.astimezone(tz)
    weekdays = set(market.get("trading_weekdays", [6, 0, 1, 2, 3]))
    holidays_file = PRICES.parent / "holidays.txt"
    holidays: set[date] = set()
    if holidays_file.exists():
        for line in holidays_file.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                try:
                    holidays.add(date.fromisoformat(value))
                except ValueError:
                    continue
    day = local_now.date()
    if not (local_now.hour >= int(market.get("session_close_hour", 15))
            and day.weekday() in weekdays and day not in holidays):
        day -= timedelta(days=1)
    for _ in range(32):
        if day.weekday() in weekdays and day not in holidays:
            return day
        day -= timedelta(days=1)
    return day


# ---------------------------------------------------------------- التحديث الكامل
def refresh_all(cfg: dict, codes: list[str], state: dict, offline: bool = False,
               now: datetime | None = None) -> pd.DataFrame:
    """يحدّث الكاش لكل الرموز ويرجع جدول التغطية."""
    d = cfg["data"]
    market_now = now or datetime.now(ZoneInfo(cfg.get("market", {}).get("timezone", "Africa/Cairo")))
    expected_session = expected_latest_session(market_now, cfg)
    token = _eodhd_token()
    report = []

    for i, code in enumerate(codes, 1):
        row = {"code": code, "source": "none", "bars": 0, "first_date": "", "last_date": "", "coverage_years": 0.0, "note": ""}
        cached = read_cache(code)
        fresh = None
        backfill_requested = False
        requested_start = date.today() - timedelta(days=int(d["history_years"] * 365.25))

        if not offline:
            # Initial load: request the configured multi-year history. Daily refresh:
            # request only a small overlap, then merge, so the cache retains the full
            # history without downloading it again every morning.
            backfill_state = state.setdefault("history_backfill_attempted", {})
            retry_days = max(1, int(d.get("history_backfill_retry_days", 30)))
            today = date.today()
            if cached is not None and len(cached):
                # Try to extend short history periodically, not every morning.
                # Some free sources return only ~5 years even when asked for 8;
                # repeating the same full download daily wastes time and requests.
                cache_start = cached.index[0].date()
                last_attempt = None
                try:
                    last_attempt = date.fromisoformat(str(backfill_state.get(code, "")))
                except ValueError:
                    pass
                backfill_due = last_attempt is None or (today - last_attempt).days >= retry_days
                if cache_start > requested_start + timedelta(days=30) and backfill_due:
                    start = requested_start
                    backfill_requested = True
                    backfill_state[code] = today.isoformat()
                else:
                    start = (cached.index[-1] - timedelta(days=7)).date()
                fresh = fetch_yahoo_from(code + d["yahoo_suffix"], start)
            else:
                backfill_requested = True
                fresh = fetch_yahoo(code + d["yahoo_suffix"], d["history_years"])
                # Remember if the initial free-source response cannot reach the
                # requested start, so the next daily run uses incremental refresh.
                if fresh is not None and len(fresh) and fresh.index[0].date() > requested_start + timedelta(days=30):
                    backfill_state[code] = today.isoformat()
            # A successful Yahoo response can still be shorter than requested (for
            # example, only ~5 years when 8 were requested). Use EODHD as a bounded
            # backfill fallback only on scheduled backfill attempts, respecting the
            # shared daily budget. Never discard the newer Yahoo bars.
            if (
                fresh is not None and len(fresh) and backfill_requested
                and fresh.index[0].date() > requested_start + timedelta(days=30)
                and token and _eodhd_budget(state, cfg) > 0
            ):
                state["eodhd"]["calls"] += 1
                older = fetch_eodhd(code, token, requested_start)
                if older is not None and len(older):
                    fresh = pd.concat([older, fresh])
                    fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
                    row["note"] = "استكمال التاريخ الأقدم من EODHD بعد أن أعاد Yahoo تاريخًا أقصر من المطلوب"

            time.sleep(d["request_sleep"])

        # A successful Yahoo response may still be behind the latest expected EGX
        # session even when it advances the local cache. Freshness is measured against
        # the market calendar, not merely against the previous cached candle.
        yahoo_behind_cache = (
            fresh is not None and len(fresh) > 0 and cached is not None and len(cached) > 0
            and fresh.index[-1].date() <= cached.index[-1].date()
        )
        yahoo_behind_market = fresh is not None and len(fresh) > 0 and fresh.index[-1].date() < expected_session
        yahoo_stale = yahoo_behind_cache or yahoo_behind_market
        if yahoo_stale and not offline and token and _eodhd_budget(state, cfg) > 0:
            state["eodhd"]["calls"] += 1
            if cached is not None and len(cached):
                fallback_start = (cached.index[-1] - timedelta(days=10)).date()
            else:
                fallback_start = expected_session - timedelta(days=14)
            recent = fetch_eodhd(code, token, fallback_start)
            if recent is not None and len(recent):
                fresh = pd.concat([fresh, recent]) if fresh is not None and len(fresh) else recent
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
                if fresh.index[-1].date() >= expected_session:
                    row["note"] = "Yahoo لم يقدم أحدث جلسة متوقعة؛ تم استكمال التحديث من EODHD"
                else:
                    row["note"] = (
                        f"تحذير: Yahoo وEODHD لم يوفرا الجلسة المتوقعة {expected_session}؛ "
                        f"آخر شمعة متاحة {fresh.index[-1].date()}"
                    )

        if fresh is not None and len(fresh) >= 1:
            merged = fresh
            if cached is not None and len(cached):
                merged = pd.concat([cached, fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            write_cache(code, merged)
            source = "yahoo" if cached is None else "cache+yahoo"
            if "EODHD" in row["note"] or row["note"].startswith("استكمال التاريخ الأقدم"):
                source += "+eodhd"
            row.update(source=source, bars=len(merged), first_date=str(merged.index[0].date()), last_date=str(merged.index[-1].date()), coverage_years=round((merged.index[-1] - merged.index[0]).days / 365.25, 2))
        elif cached is not None:
            merged, src = cached, "cache"
            # نحاول نكمّل آخر الشموع من EODHD لو فيه ميزانية
            if not offline and token and _eodhd_budget(state, cfg) > 0:
                state["eodhd"]["calls"] += 1
                # If Yahoo failed while the cache is short, let the fallback attempt
                # backfill the missing history instead of fetching only recent candles.
                cache_start = cached.index[0].date()
                backfill_state = state.setdefault("history_backfill_attempted", {})
                retry_days = max(1, int(d.get("history_backfill_retry_days", 30)))
                last_attempt = None
                try:
                    last_attempt = date.fromisoformat(str(backfill_state.get(code, "")))
                except ValueError:
                    pass
                backfill_due = last_attempt is None or (date.today() - last_attempt).days >= retry_days
                if cache_start > requested_start + timedelta(days=30) and backfill_due:
                    fallback_start = requested_start
                    backfill_state[code] = date.today().isoformat()
                else:
                    fallback_start = (cached.index[-1] - timedelta(days=3)).date()
                add = fetch_eodhd(code, token, fallback_start)
                if add is not None and len(add) > 0:
                    merged = pd.concat([cached, add])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    write_cache(code, merged)
                    src = "cache+eodhd"
            row.update(source=src, bars=len(merged), first_date=str(merged.index[0].date()), last_date=str(merged.index[-1].date()), coverage_years=round((merged.index[-1] - merged.index[0]).days / 365.25, 2))
            row["note"] = "Yahoo فشل، استخدمنا الكاش"
        elif not offline and token and _eodhd_budget(state, cfg) > 0:
            state["eodhd"]["calls"] += 1
            start = date.today() - timedelta(days=365)  # الخطة المجانية = سنة تاريخ
            add = fetch_eodhd(code, token, start)
            if add is not None and len(add) > 0:
                write_cache(code, add)
                row.update(source="eodhd", bars=len(add), first_date=str(add.index[0].date()), last_date=str(add.index[-1].date()), coverage_years=round((add.index[-1] - add.index[0]).days / 365.25, 2))
                row["note"] = f"من EODHD فقط ({cfg['data']['history_years']} سنوات مطلوبة؛ التغطية الفعلية حسب المصدر)"
            else:
                row["note"] = "لا بيانات (Yahoo وEODHD)"
        else:
            row["note"] = "لا بيانات"

        report.append(row)
        if i % 20 == 0:
            print(f"  ... {i}/{len(codes)}")

    return pd.DataFrame(report)


def _eodhd_token() -> str:
    return os.environ.get("EODHD_API_TOKEN") or os.environ.get("EODHD_API_KEY") or ""

def fetch_benchmark(cfg: dict, state: dict, offline: bool = False) -> pd.DataFrame | None:
    """Prefer the Sharia index for the Sharia universe; fall back to the broad EGX30 benchmark."""
    if cfg.get("universe", {}).get("mode") == "sharia_union":
        try:
            sharia = read_cache("_INDEX_EGX33_SHARIAH")
            if sharia is not None and len(sharia) > 60:
                return sharia
        except Exception:
            pass

    sym = cfg["data"].get("benchmark_yahoo_symbol", "^CASE30")
    eodhd_sym = cfg["data"].get("benchmark_eodhd_symbol", "CASE30.INDX")
    p = PRICES / "_BENCH.csv"
    if not offline:
        df = fetch_yahoo(sym, cfg["data"]["history_years"])
        if df is not None and len(df) > 60:
            df.index.name = "date"
            df[COLS].round(4).to_csv(p)
            return df
        token = _eodhd_token()
        if token and _eodhd_budget(state, cfg) > 0:
            state["eodhd"]["calls"] += 1
            try:
                add = fetch_eodhd_index(eodhd_sym, token, cfg["data"]["history_years"])
                if add is not None and len(add) > 60:
                    add[COLS].round(4).to_csv(p)
                    return add
            except Exception as e:
                print(f"  [EODHD] تعذر جلب مؤشر EGX المرجعي: {e}")
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["date"], index_col="date")[COLS]
            if len(df) > 60:
                return df
        except Exception:
            pass
    return None

def drop_incomplete_bar(df: pd.DataFrame, now: datetime, close_hour: int) -> pd.DataFrame:
    """نتجاهل شمعة اليوم لو الجلسة لسه ما خلصتش."""
    if len(df) and df.index[-1].date() >= now.date() and now.hour < close_hour:
        return df.iloc[:-1]
    return df


def load_prices(codes: list[str], cfg: dict, now: datetime) -> dict[str, pd.DataFrame]:
    out = {}
    for c in codes:
        df = read_cache(c)
        if df is None or len(df) == 0:
            continue
        df = drop_incomplete_bar(df, now, cfg["market"]["session_close_hour"])
        if len(df) >= cfg["data"]["min_bars"]:
            out[c] = df
    return out
