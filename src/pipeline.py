"""التشغيلة اليومية: فحص اليوم والوقت ← تحديث البيانات ← تحليل ← سجل التوصيات ← تقرير ← تلجرام.

كل مخرجات لوحة Streamlit بتتكتب هنا في مجلد data (ملفات JSON/CSV صغيرة تُرفع على GitHub).
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config as C
from .backtest import run_backtest, performance_warning
from .data_fetch import fetch_benchmark, fetch_market_indices, fetch_oanor_market, load_prices, refresh_all
from .report import build_report, compose_alert
from .signals import analyze, build_result, classify, latest_table
from .telegram import send_message
from .funds import build_fund_overlay, _score
from .track_record import load_history, log_signals, save_history, summarize, update_history
from .universe import build_universe
from .news import fetch_news
from .portfolio import evaluate as evaluate_portfolio, risk_summary as portfolio_risk_summary
from .portfolio_store import github_load, local_load
from .portfolio import evaluate, risk_summary
from .data_quality import apply_market_freshness_gate


class DataError(RuntimeError):
    """مشكلة في البيانات تمنع إصدار تقرير موثوق."""


def report_dir() -> Path:
    return C.DATA / "report"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ متى نشتغل؟
def should_run(cfg: dict, now: datetime, state: dict, force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "تشغيل يدوي"
    if now.weekday() not in cfg["market"]["trading_weekdays"]:
        return False, "اليوم إجازة أسبوعية (الجمعة/السبت)"
    if now.date() in C.load_holidays():
        return False, "إجازة رسمية مسجلة في data/holidays.txt"
    start, end = cfg["market"]["report_window"]
    if not (start <= now.strftime("%H:%M") <= end):
        return False, f"خارج نافذة الإرسال ({start}-{end} بتوقيت القاهرة)"
    if state.get("last_sent_date") == now.date().isoformat():
        return False, "تقرير اليوم اتبعت بالفعل"
    if state.get("report_attempt_date") == now.date().isoformat():
        return False, "تم تسجيل محاولة إرسال تقرير اليوم؛ منعًا للتكرار"
    return True, "ok"


def expected_last_session(now: datetime, cfg: dict) -> date:
    """آخر جلسة مفروض تكون بياناتها متاحة (بدون احتساب إجازات غير مسجلة)."""
    hol = C.load_holidays()
    d = now.date()
    after_close = now.hour >= cfg["market"]["session_close_hour"]
    if not (after_close and d.weekday() in cfg["market"]["trading_weekdays"] and d not in hol):
        d -= timedelta(days=1)
    for _ in range(21):
        if d.weekday() in cfg["market"]["trading_weekdays"] and d not in hol:
            return d
        d -= timedelta(days=1)
    return d


# ------------------------------------------------------------------
def build_equal_weight_index(close_series: list[pd.Series], min_constituents: int = 5) -> pd.Series | None:
    """Synthetic current-universe proxy; never forward-fill missing prices."""
    if not close_series:
        return None
    panel = pd.concat(close_series, axis=1).sort_index()
    returns = panel.pct_change(fill_method=None)
    valid_count = returns.notna().sum(axis=1)
    daily = returns.mean(axis=1, skipna=True).where(valid_count >= max(1, int(min_constituents)))
    if not daily.notna().any():
        return None
    daily = daily.loc[daily.first_valid_index():].fillna(0.0)
    # Normalize the first valid observation to 100; apply returns from the next bar.
    if len(daily):
        daily.iloc[0] = 0.0
    index = (1.0 + daily).cumprod() * 100.0
    index.name = "synthetic_sharia_equal_weight"
    return index


# ------------------------------------------------------------------ مخرجات اللوحة
def save_outputs(result: dict, latest_cls: pd.DataFrame, regime_df: pd.DataFrame, cov: pd.DataFrame, uni: pd.DataFrame, cfg: dict) -> None:
    C.SIGNALS.mkdir(parents=True, exist_ok=True)
    tbl = latest_cls.copy()
    tbl["why_not_buy"] = tbl["why_not_buy"].apply(lambda x: " | ".join(x) if isinstance(x, list) else "")
    tbl["date"] = pd.to_datetime(tbl["date"]).dt.date.astype(str)
    metadata_cols = ["code", "in_list", "in_egx33", "in_sharia_extra", "sharia_review_status", "sharia_source", "sharia_source_url", "sharia_verified_on"]
    metadata_cols = [c for c in metadata_cols if c in uni.columns]
    tbl = tbl.merge(uni[metadata_cols], on="code", how="left")
    tbl.round(4).to_csv(C.SIGNALS / "table.csv", index=False)

    reg = regime_df.tail(400).copy()
    reg.index.name = "date"
    reg.round(4).to_csv(C.SIGNALS / "regime.csv")
    # Per-symbol audit: actual history coverage joined to Sharia-universe provenance.
    coverage = cov.copy()
    if "code" in coverage.columns and "code" in uni.columns:
        universe_cols = [
            "code", "name", "in_egx33", "in_sharia_extra",
            "sharia_review_status", "sharia_source", "sharia_source_url", "sharia_verified_on",
        ]
        universe_cols = [col for col in universe_cols if col in uni.columns]
        coverage = coverage.merge(uni[universe_cols], on="code", how="left")
    if "coverage_years" in coverage.columns:
        coverage["coverage_years"] = pd.to_numeric(coverage["coverage_years"], errors="coerce")
        requested = float(cfg.get("data", {}).get("history_years", 0))
        coverage["history_status"] = "لا توجد بيانات"
        coverage.loc[coverage["coverage_years"] >= max(0.0, requested - 0.25), "history_status"] = "تغطية قريبة من المطلوب"
        coverage.loc[(coverage["coverage_years"] >= 3) & (coverage["coverage_years"] < max(0.0, requested - 0.25)), "history_status"] = "تغطية 3 سنوات فأكثر"
        coverage.loc[(coverage["coverage_years"] > 0) & (coverage["coverage_years"] < 3), "history_status"] = "تاريخ محدود"
    coverage.to_csv(C.SIGNALS / "coverage.csv", index=False)

    out = dict(result)
    out["universe"] = {
        "mode": cfg["universe"]["mode"],
        "total": int(len(uni)),
        "in_list": int(uni["in_list"].sum()),
        "in_egx33": int(uni["in_egx33"].sum()),
        "both": int((uni["in_list"] & uni["in_egx33"]).sum()),
        "sharia_verified_total": int(uni["sharia_verified"].sum()) if "sharia_verified" in uni else 0,
        "sharia_extra_outside_egx33": int((uni["in_sharia_extra"] & ~uni["in_egx33"]).sum()) if "in_sharia_extra" in uni else 0,
        "verified_outside_egx33": int((uni["sharia_verified"] & ~uni["in_egx33"]).sum()) if {"sharia_verified", "in_egx33"} <= set(uni.columns) else 0,
        "external_review_expired": int(uni["sharia_review_status"].eq("اعتماد خارجي منتهي الصلاحية").sum()) if "sharia_review_status" in uni else 0,
        "external_review_current": int(uni["sharia_review_status"].eq("اعتماد خارجي حديث").sum()) if "sharia_review_status" in uni else 0,
        "sharia_review_required": int((~uni["sharia_verified"] & ~uni["in_egx33"]).sum()) if {"sharia_verified", "in_egx33"} <= set(uni.columns) else 0,
    }
    # Expose the actual historical coverage instead of implying that every ticker
    # has the configured lookback. These values feed diagnostics and are auditable.
    if not cov.empty:
        years = pd.to_numeric(cov.get("coverage_years", pd.Series(dtype=float)), errors="coerce")
        out["history_coverage"] = {
            "requested_years": float(cfg.get("data", {}).get("history_years", 0)),
            "symbols_with_data": int((pd.to_numeric(cov.get("bars", 0), errors="coerce").fillna(0) > 0).sum()),
            "symbols_at_least_5y": int((years >= 5).sum()),
            "symbols_at_least_3y": int((years >= 3).sum()),
            "symbols_under_1y": int(((years > 0) & (years < 1)).sum()),
            "median_years": round(float(years[years > 0].median()), 2) if (years > 0).any() else 0.0,
            "oldest_first_date": str(cov.loc[cov["first_date"].astype(str).ne(""), "first_date"].min()) if "first_date" in cov and cov["first_date"].astype(str).ne("").any() else None,
            "newest_last_date": str(cov.loc[cov["last_date"].astype(str).ne(""), "last_date"].max()) if "last_date" in cov and cov["last_date"].astype(str).ne("").any() else None,
        }
    else:
        out["history_coverage"] = {"requested_years": float(cfg.get("data", {}).get("history_years", 0)), "symbols_with_data": 0, "symbols_at_least_5y": 0, "symbols_at_least_3y": 0, "symbols_under_1y": 0, "median_years": 0.0, "oldest_first_date": None, "newest_last_date": None}
    _write_json(C.SIGNALS / "latest.json", out)


def maybe_backtest(cfg: dict, state: dict, frames: dict, now: datetime, force: bool = False) -> bool:
    every = cfg["backtest"]["refresh_every_days"]
    last = state.get("last_backtest")
    due = force or not last or (now.date() - date.fromisoformat(last)).days >= every
    if not due:
        return False
    summary, strat, _base = run_backtest(frames, cfg)
    summary["generated_at"] = now.isoformat(timespec="seconds")
    summary["stocks"] = len(frames)
    _write_json(C.BACKTEST / "summary.json", summary)
    if len(strat):
        strat.to_csv(C.BACKTEST / "strategy_trades.csv", index=False)
    state["last_backtest"] = now.date().isoformat()
    return True


# ------------------------------------------------------------------ التنفيذ
def _execute(cfg: dict, now: datetime, state: dict, dry_run: bool, offline: bool, force: bool) -> dict:
    uni = build_universe(cfg)
    codes = uni["code"].tolist()
    names = dict(zip(uni["code"], uni["name"]))
    # سجل تدقيقي ثابت لكل تشغيل: ما الأسهم التي دخلت التحليل، وما مصدر اعتماد الشريعة.
    # الملف شخصي ومحلي داخل مستودع المشروع؛ لا يتضمن أسرارًا أو بيانات المحفظة.
    universe_path = C.SIGNALS / "universe.csv"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(universe_path, index=False)

    # Preserve a dated constituent/eligibility snapshot. This builds an auditable
    # point-in-time history going forward, instead of silently replacing the
    # current universe every day. It does not pretend to reconstruct old members
    # for years where no historical constituent source exists.
    snapshot_dir = C.DATA / "universe_history"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{now.date().isoformat()}.csv"
    if not snapshot_path.exists():
        snapshot = uni.copy()
        snapshot.insert(0, "snapshot_date", now.date().isoformat())
        snapshot.to_csv(snapshot_path, index=False)

    print(
        f"الأسهم المحللة: {len(codes)} "
        f"(قائمة مرجعية {int(uni['in_list'].sum())} | EGX33 {int(uni['in_egx33'].sum())} "
        f"| إضافات شريعة خارج المؤشر {int(uni['in_sharia_extra'].sum())} "
        f"| وضع الدمج: {cfg['universe']['mode']})"
    )

    # Refresh index caches first so Sharia analysis can use EGX33 as its benchmark.
    market_indices = fetch_market_indices(cfg, state, offline=offline)
    bench = fetch_benchmark(cfg, state, offline=offline)
    live_market = fetch_oanor_market() if not offline else None
    funds = build_fund_overlay(bench, cfg, state=state)
    cov = refresh_all(cfg, codes, state, offline=offline)
    prices = load_prices(codes, cfg, now)
    if len(prices) < max(5, int(0.5 * len(codes))):
        raise DataError(f"بيانات غير كافية: {len(prices)} سهم فقط من {len(codes)} وصلت بيانات كافية (Yahoo/EODHD).")

    ml_eligible_codes = set(uni.loc[uni["sharia_verified"], "code"].astype(str))
    frames, regime_df, _panel = analyze(prices, bench, cfg, eligible_ml_codes=ml_eligible_codes)
    latest = latest_table(frames, names, cfg, regime_df.attrs.get("ml_probs", {}), regime_df.attrs.get("ml_meta", {}))
    # Sharia eligibility is a hard gate for BUY labels, not merely a ranking feature.
    # Unverified candidates remain visible in WATCH with an explicit review reason.
    eligibility = uni[["code", "sharia_verified", "sharia_review_status"]].copy()
    latest = latest.merge(eligibility, on="code", how="left")
    latest["sharia_verified"] = latest["sharia_verified"].fillna(False).astype(bool)
    latest["sharia_review_status"] = latest["sharia_review_status"].fillna("مراجعة مطلوبة")
    last = regime_df.iloc[-1]
    cls = classify(latest, last["regime"], last["regime_score"], cfg)
    unverified = ~cls["sharia_verified"]
    cls.loc[unverified & cls["signal"].eq("BUY"), "signal"] = "WATCH"
    for idx in cls.index[unverified]:
        reasons = cls.at[idx, "why_not_buy"]
        if not isinstance(reasons, list):
            reasons = []
        reasons.append("لا توجد مراجعة شرعية حديثة موثقة؛ للمراقبة فقط وليس توصية شراء")
        cls.at[idx, "why_not_buy"] = reasons
    result = build_result(frames, regime_df, cls, len(codes), cfg)
    # Fail closed: a late/stale overall market snapshot must never produce fresh BUYs.
    exp_session = expected_last_session(now, cfg)
    cls, freshness_blocked = apply_market_freshness_gate(cls, result["data_date"], exp_session)
    if freshness_blocked:
        result = build_result(frames, regime_df, cls, len(codes), cfg)
        print(f"  [DATA QUALITY] تم إيقاف الشراء: البيانات {result['data_date']} أقدم من الجلسة المتوقعة {exp_session}.")
    sharia_codes = uni.loc[uni["sharia_verified"], "code"].tolist()
    sharia_parts = [prices[c]["close"].rename(c) for c in sharia_codes if c in prices]
    # Proxy only: current constituents are not historical index membership.
    # Missing prices are not forward-filled, preventing artificial zero returns.
    sharia_close = build_equal_weight_index(sharia_parts, min_constituents=5)
    result["funds"] = funds
    if sharia_close is not None:
        result["funds"]["sharia_equity"].update(_score(sharia_close))
        result["funds"]["sharia_equity"].update({
            "reference": "EGX33 Shariah proxy",
            "reference_method": "equal_weight_current_verified_universe",
            "historical_constituent_membership": "not_available",
        })
    result["market_indices"] = market_indices
    result["live_market"] = live_market
    # المحفظة مشفرة؛ لا تُحفظ بيانات المالك أو الكميات كنص صريح داخل المستودع.
    try:
        holdings, synced = github_load()
        if not synced:
            holdings = local_load()
        if holdings:
            ptable = cls.copy()
            ptable["code"] = ptable["code"].astype(str)
            pdf = evaluate_portfolio(holdings, ptable, funds, result["regime"]["label"], max_single_position_pct=float(cfg.get("portfolio", {}).get("max_single_position_pct", 25)))
            result["portfolio"] = {
                "summary": portfolio_risk_summary(pdf),
                "decisions": pdf[["النوع", "الأصل", "الكمية", "متوسط التكلفة", "السعر الحالي", "الربح/الخسارة %", "النقاط", "القرار", "reason"]].where(pd.notna(pdf), None).to_dict("records") if not pdf.empty else [],
                "encrypted_sync": synced,
            }
    except Exception as e:
        print(f"  [PORTFOLIO] تعذر تحميل المحفظة: {type(e).__name__}")
    try:
        result["news"] = fetch_news(result["picks"]) if not offline else {"market": [], "companies": [], "offline": True}
    except Exception as e:
        print(f"  [NEWS] تعذر جلب الأخبار: {type(e).__name__}")
        result["news"] = {"market": [], "companies": [], "error": type(e).__name__}
    # تحليل المحفظة المحفوظة (إن وُجدت) حتى يدخل في تقرير الصباح تلقائيًا.
    try:
        from .portfolio_store import load_for_daily
        holdings = load_for_daily()
        result["portfolio"] = {"holdings": holdings, "enabled": bool(holdings)}
        if holdings:
            pdf = evaluate(holdings, latest, funds, result["regime"]["label"], float(cfg.get("portfolio", {}).get("max_single_position_pct", 25)))
            result["portfolio"]["summary"] = risk_summary(pdf)
            result["portfolio"]["decisions"] = pdf.to_dict("records")
    except Exception as e:
        print(f"  [PORTFOLIO] تعذر تحميل المحفظة: {type(e).__name__}")
        result["portfolio"] = {"holdings": [], "enabled": False, "error": type(e).__name__}
    result["generated_at"] = now.isoformat(timespec="seconds")

    # ---- ملاحظات جودة البيانات
    warnings: list[str] = []
    data_d = date.fromisoformat(result["data_date"])
    exp = expected_last_session(now, cfg)
    if data_d < exp:
        warnings.append(f"إيقاف آمن: آخر جلسة في البيانات {data_d} أقدم من الجلسة المتوقعة {exp}؛ تم تحويل كل توصيات الشراء إلى مراقبة لحين تحديث البيانات. تحقق أيضًا من data/holidays.txt.")
    missing = [c for c in codes if c not in prices]
    if missing:
        warnings.append(f"{len(missing)} سهم بدون بيانات كافية (منها: {', '.join(missing[:6])}).")
    # Requested lookback is not the same as actual coverage; report the gap explicitly.
    if "coverage_years" in cov.columns and len(cov):
        coverage = pd.to_numeric(cov["coverage_years"], errors="coerce").fillna(0.0)
        requested_years = float(cfg["data"].get("history_years", 5))
        short_history = int((coverage < max(0.0, requested_years - 0.25)).sum())
        if short_history:
            warnings.append(
                f"التغطية التاريخية: {short_history} من {len(cov)} رمزًا أقل من "
                f"{requested_years:g} سنوات المطلوبة؛ الاختبار التاريخي لا يستخدم إلا الفترات المتاحة فعليًا لكل سهم."
            )
    n_cache = int(cov["source"].astype(str).str.startswith("cache").sum())
    if n_cache:
        warnings.append(f"{n_cache} سهم اتحلل ببيانات محفوظة لأن Yahoo فشل معها.")
    if bench is None or len(bench) < 60:
        warnings.append("لم تتوفر سلسلة EGX30 التاريخية من المزود؛ التحليل استخدم مؤشرًا داخليًا متساوي الأوزان من الأسهم المحللة.")
    if sharia_close is not None:
        warnings.append("مرجع الأسهم الشرعية بديل حسابي متساوي الأوزان من المكونات الحالية ذات البيانات المتاحة، وليس السجل الرسمي التاريخي لمكونات EGX33؛ قد يتأثر بانحياز بقاء الشركات.")
    if not result.get("news", {}).get("market"):
        warnings.append("مصدر الأخبار غير متاح؛ لم تُستخدم الأخبار في أي قرار رقمي.")
    n_stale = int(result["signal_counts"].get("STALE", 0))
    if n_stale:
        warnings.append(f"{n_stale} سهم بيانات آخر شمعة له قديمة (موقوف أو متأخر).")

    # شرعية السهم ليست صفة دائمة؛ اعرض عمر مصدر الفرز بدل افتراض استمراره.
    if "sharia_verified_on" in uni.columns:
        extra_dates = pd.to_datetime(
            uni.loc[uni["in_sharia_extra"].fillna(False), "sharia_verified_on"],
            errors="coerce",
        ).dropna()
        extra_count = int(uni["in_sharia_extra"].fillna(False).sum())
        if extra_count and len(extra_dates) < extra_count:
            warnings.append(
                f"مراجعة الشريعة: {extra_count - len(extra_dates)} سهم خارج EGX33 بلا تاريخ تحقق؛ راجع data/sharia_extra.csv."
            )
        if len(extra_dates):
            latest_review = extra_dates.max().date()
            review_age = (now.date() - latest_review).days
            if review_age > 120:
                warnings.append(
                    f"قائمة الشريعة للأسهم خارج EGX33 قديمة نسبيًا: أحدث تاريخ تحقق {latest_review} "
                    f"(منذ {review_age} يومًا). الأسهم تظل في التحليل لكن يلزم إعادة التحقق من المصدر الشرعي قبل الاعتماد عليها."
                )

    # A stale run is diagnostic-only. Do not replace the last published report,
    # append stale signals to the track record, or advance any daily state.
    # Otherwise a failed provider refresh can make an old snapshot look newly generated.
    if data_d < exp:
        diagnostic = {
            "status": "blocked_stale_data",
            "generated_at": now.isoformat(timespec="seconds"),
            "data_date": data_d.isoformat(),
            "expected_session": exp.isoformat(),
            "symbols_with_data": int(len(prices)),
            "universe_size": int(len(codes)),
            "coverage_rows": int(len(cov)),
            "source_counts": cov["source"].astype(str).value_counts().to_dict() if "source" in cov else {},
            "warnings": warnings,
        }
        _write_json(C.DATA / "diagnostics" / "last_run.json", diagnostic)
        print(f"  [DATA QUALITY] لم يتم تحديث التقرير المنشور: البيانات {data_d} أقدم من الجلسة المتوقعة {exp}.")
        return {
            "status": "not_sent",
            "detail": f"توقف آمن: بيانات آخر جلسة {data_d} أقدم من الجلسة المتوقعة {exp}؛ احتُفظ بآخر تقرير منشور ولم تُسجّل توصيات جديدة.",
            "data_date": result["data_date"],
            "expected_session": exp.isoformat(),
            "regime": result["regime"]["label_ar"],
            "buys": [],
            "warnings": warnings,
            "diagnostic": "data/diagnostics/last_run.json",
        }

    # ---- سجل التوصيات
    hist = load_history()
    hist = update_history(hist, prices, cfg)
    hist = log_signals(hist, result["picks"], result["data_date"], result["regime"]["label"])
    save_history(hist)
    track = summarize(hist)

    save_outputs(result, cls, regime_df, cov, uni, cfg)
    ran_bt = maybe_backtest(cfg, state, frames, now, force=False)
    if ran_bt:
        print("تم تحديث نتائج الاختبار التاريخي.")
    # Always expose weak historical evidence, even when the weekly backtest
    # refresh is not due. Do not hide a poor comparison behind fresh signals.
    try:
        bt_path = C.BACKTEST / "summary.json"
        if bt_path.exists():
            bt_summary = json.loads(bt_path.read_text(encoding="utf-8"))
            bt_warning = performance_warning(bt_summary)
            if bt_warning:
                warnings.append(bt_warning)
    except Exception as e:
        print(f"  [BACKTEST] تعذر تقييم ملخص الأداء: {type(e).__name__}")

    # ---- التقرير
    rep = build_report(result, track, cfg, now, warnings)
    rep["track_record"] = track
    _write_json(report_dir() / "latest.json", rep)
    print(f"الذكاء الاصطناعي: {rep['llm']}")

    status = "generated"
    detail = ""
    # A stale report may be useful for diagnostics, but must never be sent or
    # published as a fresh daily recommendation. Fail closed until data catches up.
    if data_d < exp:
        status = "not_sent"
        detail = f"توقف آمن: بيانات آخر جلسة {data_d} أقدم من الجلسة المتوقعة {exp}؛ لم يتم إرسال أو نشر توصيات."
    elif cfg["telegram"]["enabled"]:
        if dry_run:
            status = "dry_run"
        else:
            token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat:
                status, detail = "not_sent", "TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID غير مضبوط في الـ Secrets."
            else:
                # Reserve before Telegram to avoid duplicate delivery after interruption.
                # Ambiguous failures require manual review; retries are blocked for this date.
                state["report_attempt_date"] = now.date().isoformat()
                state["report_attempt_at"] = now.isoformat(timespec="seconds")
                C.save_state(state)
                n = send_message(token, chat, rep["telegram_html"])
                state["last_sent_date"] = now.date().isoformat()
                status, detail = "sent", f"{n} رسالة"
    return {
        "status": status,
        "detail": detail,
        "data_date": result["data_date"],
        "regime": result["regime"]["label_ar"],
        "buys": [p["code"] for p in result["picks"]],
        "warnings": warnings,
    }


def _alert(cfg: dict, reason: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if cfg["telegram"]["enabled"] and token and chat:
        try:
            send_message(token, chat, compose_alert(reason))
        except Exception as e:  # noqa: BLE001
            print(f"تعذر إرسال تنبيه الفشل: {type(e).__name__}")


def run(cfg: dict | None = None, force: bool = False, dry_run: bool = False, offline: bool = False,
        now: datetime | None = None, notify_errors: bool = True) -> dict:
    cfg = cfg or C.load_config()
    now = now or C.cairo_now(cfg)
    state = C.load_state()
    ok, why = should_run(cfg, now, state, force)
    if not ok:
        print(f"تخطي التشغيل: {why}")
        return {"status": "skipped", "detail": why}
    try:
        return _execute(cfg, now, state, dry_run, offline, force)
    except Exception as e:  # noqa: BLE001
        if notify_errors and not dry_run:
            _alert(cfg, f"{type(e).__name__}: {e}")
        raise
    finally:
        C.save_state(state)
