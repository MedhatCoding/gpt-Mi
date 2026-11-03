"""اختبارات بدون إنترنت (بيانات وهمية). التشغيل:  python tests/test_engine.py   أو   pytest tests

ملحوظة: نحدد EGX_DATA_DIR قبل استيراد أي شيء من src، لأن المسارات بتتحدد وقت الاستيراد.
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="egx_test_"))
os.environ["EGX_DATA_DIR"] = str(TMP)
# Keep the offline suite deterministic and avoid consuming personal Gemini/API quota.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tests.synth import make_prices  # noqa: E402

from src import config as C  # noqa: E402
from src import pipeline, report, telegram  # noqa: E402
from src.backtest import performance_warning, run_backtest  # noqa: E402
from src.indicators import atr, compute_indicators, rsi, sma  # noqa: E402
from src.ml_model import _dataset, train_predict  # noqa: E402
from src.scoring import compute_scores  # noqa: E402
from src.regime import compute_regime  # noqa: E402
from src.signals import analyze, build_result, classify, latest_table  # noqa: E402
from src.sim import simulate_trade  # noqa: E402
from src.track_record import COMPONENTS, component_stats, load_history, log_signals, save_history, summarize, update_history  # noqa: E402
from src.universe import build_universe  # noqa: E402
from src.funds import SHARIA_FUNDS, GOLD_FUNDS, FUND_SOURCE_MATCHES  # noqa: E402

CFG = C.load_config()
CAIRO = ZoneInfo("Africa/Cairo")
SUN_9AM = datetime(2026, 9, 20, 9, 5, tzinfo=CAIRO)


# ------------------------------------------------------------------ تجهيز البيانات
def _bull_prices(n: int = 50) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    return {
        f"B{i:03d}": make_prices(
            100 + i, drift=float(rng.choice([0.0015, 0.001, 0.0006, 0.0002])), vol=float(rng.uniform(0.012, 0.022)),
            start_price=float(rng.uniform(5, 80)),
        )
        for i in range(n)
    }


def _setup_data_dir() -> list[str]:
    (TMP / "prices").mkdir(parents=True, exist_ok=True)
    codes = []
    for c, df in _bull_prices().items():
        df.to_csv(TMP / "prices" / f"{c}.csv")
        codes.append(c)
    make_prices(999, drift=0.001, base_volume=2_000, start_price=10).to_csv(TMP / "prices" / "ILLQ.csv")
    make_prices(998, drift=0.001, end="2026-08-20").to_csv(TMP / "prices" / "STAL.csv")
    make_prices(997, n=90).to_csv(TMP / "prices" / "SHRT.csv")
    listed = codes[:45] + ["ILLQ", "STAL", "SHRT"]
    pd.DataFrame({"code": listed, "name_ar": [f"شركة {c}" for c in listed]}).to_csv(TMP / "universe.csv", index=False)
    (TMP / "egx33.txt").write_text("# test\n" + "\n".join(f"{c},Company {c}" for c in codes[40:]) + "\n", encoding="utf-8")
    # Synthetic fixtures explicitly screened as Sharia-compliant so the production
    # sharia_union gate can be tested without weakening eligibility in test code.
    # Include the illiquid/stale/short-history test symbols in the approved
    # synthetic universe so the end-to-end pipeline can assert their exclusion
    # reasons. Their presence here is test-only, not a real-world Sharia claim.
    screened_codes = codes + ["ILLQ", "STAL", "SHRT"]
    pd.DataFrame({
        "code": screened_codes,
        "name_ar": [f"شركة {c}" for c in screened_codes],
        "verification_source": ["test fixture: approved screening"] * len(screened_codes),
        "verified_on": ["2026-09-01"] * len(screened_codes),
        "notes": ["synthetic test data"] * len(screened_codes),
        "status": ["approved"] * len(screened_codes),
    }).to_csv(TMP / "sharia_extra.csv", index=False)
    (TMP / "holidays.txt").write_text("# none\n2026-10-06\n", encoding="utf-8")
    return codes


def _reset_outputs() -> None:
    for name in ("signals", "report", "backtest"):
        shutil.rmtree(TMP / name, ignore_errors=True)
    (TMP / "state.json").unlink(missing_ok=True)


CODES = _setup_data_dir()


def _analysis(prices=None):
    prices = prices or {c: p for c, p in _bull_prices().items()}
    frames, regime_df, _ = analyze(prices, None, CFG)
    latest = latest_table(frames, {c: c for c in frames}, CFG)
    # Test universe is explicitly approved by _setup_data_dir().
    latest["sharia_verified"] = True
    latest["sharia_review_status"] = "اعتماد اختبار حديث"
    last = regime_df.iloc[-1]
    cls = classify(latest, last["regime"], last["regime_score"], CFG)
    result = build_result(frames, regime_df, cls, len(prices), CFG)
    return frames, regime_df, cls, result


# ------------------------------------------------------------------ المؤشرات
def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(10, 20, 60))
    assert rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.linspace(20, 10, 60))
    assert rsi(down).iloc[-1] < 1.0
    r = rsi(make_prices(1)["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_sma_and_atr():
    s = pd.Series(np.arange(1.0, 101.0))
    assert abs(sma(s, 20).iloc[-1] - s.iloc[-20:].mean()) < 1e-9
    df = make_prices(2)
    a = atr(df["high"], df["low"], df["close"]).dropna()
    assert (a > 0).all()


def test_indicators_are_causal():
    """قيمة المؤشر والنقاط في يوم معين لازم ماتتغيرش لو حذفنا بيانات المستقبل (لا تسريب مستقبل)."""
    df = make_prices(11, drift=0.001)
    full = compute_scores(compute_indicators(df), None, None, CFG)
    for i in (300, 420, 560):
        part = compute_scores(compute_indicators(df.iloc[: i + 1]), None, None, CFG)
        for col in ("score", "stop", "t1", "rsi14", "atr14", "sma50", "high20", "buy_raw"):
            a, b = full[col].iloc[i], part[col].iloc[-1]
            assert (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) < 1e-9, (col, i, a, b)



def test_seasonality_has_no_future_leakage():
    """تغيير السنوات المستقبلية لا يغير درجة الموسمية التاريخية."""
    idx = pd.bdate_range("2015-01-01", "2025-12-31")
    rng = np.random.default_rng(731)
    returns = rng.normal(0.0003, 0.012, len(idx))
    bench = pd.Series(100 * np.cumprod(1 + returns), index=idx)
    panel = pd.DataFrame({"SYN": bench}, index=idx)
    changed = bench.copy()
    future = changed.index >= pd.Timestamp("2024-01-01")
    changed.loc[future] *= np.linspace(1.0, 2.0, int(future.sum()))
    changed_panel = pd.DataFrame({"SYN": changed}, index=idx)
    original_regime = compute_regime(bench, panel, CFG)
    changed_regime = compute_regime(changed, changed_panel, CFG)
    cutoff = pd.Timestamp("2023-12-29")
    a = original_regime.loc[:cutoff, "seasonality_score"]
    b = changed_regime.loc[:cutoff, "seasonality_score"]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    # Ensure the fixture includes dates where historical seasonal observations exist.
    assert original_regime.loc["2023-06-01":"2023-06-30", "seasonality_month_return"].notna().any()


# ------------------------------------------------------------------ المحاكاة
def _path(o, h, l, c):
    return [np.array(x, dtype=float) for x in (o, h, l, c)]


def test_sim_target_stop_gap_and_skip():
    # فهرس 0 = شمعة الإشارة. الدخول افتتاح الشمعة 1 = 100. الوقف 95 والهدف 110
    o, h, l, c = _path([100, 100, 101, 102], [101, 104, 111, 103], [99, 99, 100, 101], [100, 103, 110, 102])
    r = simulate_trade(o, h, l, c, 0, 95, 110, 20, 0.5)
    assert r["status"] == "closed" and r["reason"] == "target" and r["exit"] == 110
    # الوقف والهدف في نفس الشمعة = الوقف الأول
    o, h, l, c = _path([100, 100, 100], [101, 112, 100], [99, 94, 99], [100, 100, 100])
    assert simulate_trade(o, h, l, c, 0, 95, 110, 20, 0.5)["reason"] == "stop"
    # فجوة تحت الوقف: الخروج بالافتتاح
    o, h, l, c = _path([100, 100, 90], [101, 101, 92], [99, 99, 89], [100, 100, 91])
    r = simulate_trade(o, h, l, c, 0, 95, 110, 20, 0.5)
    assert r["reason"] == "stop" and r["exit"] == 90
    # الدخول تحت الوقف أو فوق الهدف = تجاهل
    o, h, l, c = _path([100, 94], [101, 96], [99, 93], [100, 95])
    assert simulate_trade(o, h, l, c, 0, 95, 110, 20, 0.5)["status"] == "skipped"
    # لا شمعة تالية بعد = pending
    assert simulate_trade(*_path([100], [101], [99], [100]), 0, 95, 110, 20, 0.5)["status"] == "pending"


def test_sim_time_exit():
    n = 30
    o = np.full(n, 100.0); h = np.full(n, 101.0); l = np.full(n, 99.0); c = np.full(n, 100.0)
    r = simulate_trade(o, h, l, c, 0, 95, 110, 10, 0.5)
    assert r["status"] == "closed" and r["reason"] == "time" and r["days"] == 10


# ------------------------------------------------------------------ الإشارات وحالة السوق
def test_seasonality_has_no_future_data_leakage():
    """قيمة الموسمية عند تاريخ قديم لا تتغير عند إضافة بيانات مستقبلية."""
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    t = np.arange(len(dates), dtype=float)
    monthly_wave = np.sin(2 * np.pi * pd.Index(dates).month.to_numpy() / 12.0)
    close = 1000.0 * np.exp(0.00015 * t + 0.0008 * np.cumsum(monthly_wave))
    bench = pd.Series(close, index=dates, name="close")
    panel = pd.DataFrame(
        {f"S{i}": close * (1.0 + i * 0.002) for i in range(5)},
        index=dates,
    )
    cutoff = pd.Timestamp("2022-06-15")
    full = compute_regime(bench, panel, CFG)
    truncated = compute_regime(bench.loc[:cutoff], panel.loc[:cutoff], CFG)
    check_date = pd.Timestamp("2022-06-15")
    assert np.isclose(
        float(full.loc[check_date, "seasonality_score"]),
        float(truncated.loc[check_date, "seasonality_score"]),
        atol=1e-12,
    )
    assert np.isclose(
        float(full.loc[check_date, "regime_score"]),
        float(truncated.loc[check_date, "regime_score"]),
        atol=1e-12,
    )


def test_bull_universe_gives_valid_picks():
    frames, regime_df, cls, result = _analysis()
    assert result["regime"]["label"] in ("up", "strong_up")
    picks = result["picks"]
    assert 0 < len(picks) <= CFG["risk"]["max_buys_by_regime"][result["regime"]["label"]]
    for p in picks:
        assert p["score"] >= CFG["scoring"]["buy_score"]
        assert p["stop"] < p["close"] < p["t1"] < p["t2"]
        assert p["entry_low"] < p["entry_high"]
        assert p["stop"] < p["entry_low"]
        assert 0 < p["position_pct"] <= CFG["risk"]["max_position_pct"]
    assert result["market"]["advancers"] + result["market"]["decliners"] > 0


def test_bear_universe_gives_no_buys():
    rng = np.random.default_rng(5)
    prices = {f"D{i:03d}": make_prices(500 + i, drift=float(rng.choice([-0.0012, -0.0008, -0.0004])), vol=0.015, start_price=50) for i in range(40)}
    _, _, cls, result = _analysis(prices)
    assert result["regime"]["label"] in ("weak", "down")
    assert len(result["picks"]) <= CFG["risk"]["max_buys_by_regime"][result["regime"]["label"]]
    if result["regime"]["label"] == "down":
        assert result["picks"] == []


def test_limit_down_forces_avoid_and_blocks_buy():
    """سهم هبط بالحد الأدنى في آخر جلسة: يُستبعد من الشراء ويتصنّف AVOID حتى لو نقاطه عالية."""
    df = make_prices(321, drift=0.0015, vol=0.014, start_price=40)  # اتجاه صاعد قوي عشان نقاطه تبقى عالية أصلاً
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-2] * (1 - 0.098)
    df.iloc[-1, df.columns.get_loc("low")] = df["close"].iloc[-1]
    df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-2]

    prices = _bull_prices(30)
    prices["LDWN"] = df
    frames, regime_df, cls, result = _analysis(prices)
    row = cls[cls["code"] == "LDWN"].iloc[0]
    assert bool(row["limit_down"]) is True
    assert row["signal"] == "AVOID"
    assert "LDWN" not in [p["code"] for p in result["picks"]]
    assert "LDWN" not in [w["code"] for w in result["watch"]]


def test_near_limit_caution_appears_for_watch_or_buy():
    frames, regime_df, cls, result = _analysis()
    for p in result["picks"] + result["watch"]:
        row = cls[cls["code"] == p["code"]].iloc[0]
        if bool(row["near_limit"]):
            assert any("حد التداول" in c for c in p["cautions"])


def test_illiquid_and_stale_are_excluded():
    prices = _bull_prices(30)
    prices["ILLQ"] = make_prices(999, drift=0.002, base_volume=2_000, start_price=10)
    prices["STAL"] = make_prices(998, drift=0.002, end="2026-08-20")
    _, _, cls, result = _analysis(prices)
    sig = dict(zip(cls["code"], cls["signal"]))
    assert sig["ILLQ"] == "ILLIQUID"
    assert sig["STAL"] == "STALE"
    assert "ILLQ" not in [p["code"] for p in result["picks"]]


# ------------------------------------------------------------------ الاختبار التاريخي وسجل التوصيات
def test_backtest_runs_and_no_overlap_per_symbol():
    frames, *_ = _analysis()
    summary, strat, base = run_backtest(frames, CFG)
    assert summary["strategy"].get("trades", 0) > 0 and summary["baseline"]["trades"] > 0
    sensitivity = summary["cost_sensitivity"]["scenarios"]
    assert [x["additional_friction_pct"] for x in sensitivity] == [0.25, 0.50]
    assert sensitivity[0]["total_round_trip_cost_pct"] > summary["cost_sensitivity"]["base_round_trip_cost_pct"]
    assert sensitivity[1]["total_round_trip_cost_pct"] > sensitivity[0]["total_round_trip_cost_pct"]
    # Higher friction must not improve average net return when the same trades are replayed.
    base_avg = summary["strategy"]["avg_ret_pct"]
    stress_1 = sensitivity[0]["strategy"]["avg_ret_pct"]
    stress_2 = sensitivity[1]["strategy"]["avg_ret_pct"]
    if base_avg is not None and stress_1 is not None and stress_2 is not None:
        assert stress_1 < base_avg
        assert stress_2 < stress_1
    for _, g in strat.groupby("code"):
        g = g.sort_values("entry_date")
        assert (pd.to_datetime(g["entry_date"].iloc[1:].values) > pd.to_datetime(g["exit_date"].iloc[:-1].values)).all()


def test_track_record_logging_is_idempotent_and_resolves():
    _reset_outputs()
    frames, regime_df, cls, result = _analysis()
    prices = {c: f[["open", "high", "low", "close", "volume"]] for c, f in frames.items()}
    # نرجّع الزمن يومين لورا عشان تبقى فيه شموع بعد الإشارة
    cut = {c: p.iloc[:-6] for c, p in prices.items()}
    hist = load_history()
    picks = result["picks"][:3]
    date_ = str(cut[picks[0]["code"]].index[-1].date())
    hist = log_signals(hist, picks, date_, "up")
    hist2 = log_signals(hist, picks, date_, "up")
    assert len(hist) == len(hist2) == len(picks)          # تسجيل نفس اليوم مرتين لا يكرر
    hist = update_history(hist, prices, CFG)              # دلوقتي فيه شموع بعد الإشارة
    assert set(hist["status"]) <= {"closed", "open", "skipped", "pending"}
    assert summarize(hist)["recommendations"] == len(picks)


# ------------------------------------------------------------------ متى يشتغل التقرير
def test_component_stats_needs_min_trades_and_finds_real_separation():
    n_lo, n_hi = 12, 12
    rows = []
    for i in range(n_lo):
        rows.append({"status": "closed", "r_net": -0.3 + 0.01 * i, "comp_trend": 0.2, "comp_momentum": 0.5,
                     "comp_rs": 0.5, "comp_volume": 0.5, "comp_setup": 0.5})
    for i in range(n_hi):
        rows.append({"status": "closed", "r_net": 1.2 + 0.01 * i, "comp_trend": 0.9, "comp_momentum": 0.5,
                     "comp_rs": 0.5, "comp_volume": 0.5, "comp_setup": 0.5})
    closed = pd.DataFrame(rows)
    assert component_stats(closed.iloc[:10]) is None                     # عينة أصغر من الحد الأدنى
    stats = component_stats(closed)
    assert stats is not None and "comp_trend" in stats
    ct = stats["comp_trend"]
    assert ct["high_avg_r"] > ct["low_avg_r"] and ct["high_n"] == n_hi and ct["low_n"] == n_lo
    # مكوّن بدون تباين حقيقي بين النصفين ما يظهرش أصلاً هنا لأن القيم متساوية في كل الصفوف
    assert "comp_momentum" not in stats or abs(stats["comp_momentum"]["high_avg_r"] - stats["comp_momentum"]["low_avg_r"]) < 1e-9


def test_track_record_carries_components_through_pipeline():
    _reset_outputs()
    frames, regime_df, cls, result = _analysis()
    picks = result["picks"][:2]
    for p in picks:
        for c in COMPONENTS:
            assert c in p and p[c] is not None
    hist = log_signals(load_history(), picks, "2026-09-17", "up")
    for c in COMPONENTS:
        assert hist[c].notna().all()


def test_should_run_rules():
    st = {}
    assert pipeline.should_run(CFG, SUN_9AM, st)[0]
    assert not pipeline.should_run(CFG, datetime(2026, 9, 18, 9, 5, tzinfo=CAIRO), st)[0]      # جمعة
    assert not pipeline.should_run(CFG, datetime(2026, 9, 19, 9, 5, tzinfo=CAIRO), st)[0]      # سبت
    assert not pipeline.should_run(CFG, datetime(2026, 10, 6, 9, 5, tzinfo=CAIRO), st)[0]      # إجازة مسجلة
    assert not pipeline.should_run(CFG, datetime(2026, 9, 20, 7, 0, tzinfo=CAIRO), st)[0]      # بدري
    assert not pipeline.should_run(CFG, datetime(2026, 9, 20, 13, 0, tzinfo=CAIRO), st)[0]     # متأخر
    assert not pipeline.should_run(CFG, SUN_9AM, {"last_sent_date": "2026-09-20"})[0]          # اتبعت
    assert pipeline.should_run(CFG, SUN_9AM, {"last_sent_date": "2026-09-20"}, force=True)[0]


def test_expected_last_session():
    assert pipeline.expected_last_session(SUN_9AM, CFG).isoformat() == "2026-09-17"            # الأحد ← الخميس
    assert pipeline.expected_last_session(datetime(2026, 9, 21, 9, 0, tzinfo=CAIRO), CFG).isoformat() == "2026-09-20"


# ------------------------------------------------------------------ التقرير والذكاء الاصطناعي
def test_number_guard():
    al = report.allowed_numbers({"a": 3.3632, "b": 84.0, "c": [22.5367]})
    assert report.numbers_ok("مخاطرة 3.4% وقوة 84 وسعر 22.54", al)
    assert report.numbers_ok("المتوسط ٥٠ يومًا", al)
    assert not report.numbers_ok("الهدف 250 جنيه", al)
    assert not report.numbers_ok("نمو 1500", al)


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._b, self.text = status, body, text

    def json(self):
        return self._b


def test_llm_notes_are_validated():
    _, _, _, result = _analysis()
    p0 = result["picks"][0]["code"]
    good = {"market_view": "السوق داعم بحذر.", "picks": {p0: "اتجاه قوي، والخطر هو التشبع."}, "watch_view": "تابع القائمة."}
    bad = {"market_view": "توقعات بارتفاع 4321 نقطة", "picks": {p0: "هدف 999.99 مضمون"}, "watch_view": "ok"}
    payload = report.build_payload(result, None)
    fb = report.fallback_notes(result)

    def fake(body):
        return lambda *a, **k: _Resp(200, {"content": [{"type": "text", "text": body}]})

    import json

    env = {"ANTHROPIC_API_KEY": "k"}
    cfg_an = {**CFG, "llm": {**CFG["llm"], "provider": "anthropic"}}
    data, meta = report.call_llm(payload, cfg_an, post=fake(json.dumps(good, ensure_ascii=False)), env=env)
    assert meta["ok"] and data["market_view"].startswith("السوق")
    merged, dropped = report.merge_notes(data, fb, payload)
    assert merged["market_view"] == good["market_view"] and merged["picks"][p0] == good["picks"][p0] and dropped == 0

    data, meta = report.call_llm(payload, cfg_an, post=fake("```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```"), env=env)
    merged, dropped = report.merge_notes(data, fb, payload)
    assert dropped == 2 and merged["market_view"] == fb["market_view"] and merged["picks"][p0] == fb["picks"][p0]

    # فشل الخدمة أو مفتاح ناقص = رجوع للنص الجاهز بدون كسر
    assert report.call_llm(payload, cfg_an, post=lambda *a, **k: _Resp(500), env=env)[0] is None
    assert report.call_llm(payload, cfg_an, post=fake("مش JSON"), env=env)[0] is None
    assert report.call_llm(payload, CFG, env={})[0] is None

    gem = {"candidates": [{"content": {"parts": [{"text": json.dumps(good, ensure_ascii=False)}]}}]}
    cfg2 = {**CFG, "llm": {**CFG["llm"], "provider": "gemini"}}
    data, meta = report.call_llm(payload, cfg2, post=lambda *a, **k: _Resp(200, gem), env={"GEMINI_API_KEY": "k"})
    assert meta["ok"] and data["watch_view"] == good["watch_view"]


def test_message_content_and_size():
    _, _, _, result = _analysis()
    now = SUN_9AM
    rep = report.build_report(result, None, CFG, now, ["ملاحظة تجريبية"], env={})
    msg = rep["telegram_html"]
    for p in result["picks"][: CFG["telegram"]["max_picks"]]:
        assert p["code"] in msg
    assert "الوقف" in msg and "الهدف 1" in msg and "ليس نصيحة استثمارية" in msg and "ملاحظة تجريبية" in msg
    assert all(len(c) <= 4096 for c in telegram.split_message(msg))
    assert "<script" not in report.build_report({**result, "picks": [{**result["picks"][0], "name": "<script>x</script>"}]}, None, CFG, now, [], env={})["telegram_html"]


# ------------------------------------------------------------------ تلجرام
def test_telegram_split_and_fallback_and_errors():
    text = "\n\n".join(f"فقرة {i} " + "س" * 900 for i in range(15))
    chunks = telegram.split_message(text)
    assert len(chunks) > 1 and all(len(c) <= telegram.LIMIT for c in chunks)

    calls = []

    def post(url, json=None, timeout=None):
        calls.append(json)
        if json.get("parse_mode") == "HTML" and len(calls) == 1:
            return _Resp(400, text="Bad Request: can't parse entities")
        return _Resp(200)

    assert telegram.send_message("TOKEN", "1", "<b>مرحبا</b>", post=post, pause=0) == 1
    assert calls[1]["text"] == "مرحبا" and "parse_mode" not in calls[1]

    def boom(url, json=None, timeout=None):
        return _Resp(401, text="Unauthorized")

    try:
        telegram.send_message("SECRET-TOKEN-123", "1", "x", post=boom, pause=0)
        raise AssertionError("كان لازم يفشل")
    except RuntimeError as e:
        assert "SECRET-TOKEN-123" not in str(e) and "401" in str(e)


# ------------------------------------------------------------------ التشغيل الكامل
def _run(**kw):
    sent = []
    pipeline.send_message = lambda token, chat, text: (sent.append(text) or 1)
    os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"] = "t", "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    # Exercise the production Sharia-only universe with explicitly screened
    # synthetic fixtures; do not bypass the eligibility gate in end-to-end tests.
    return pipeline.run(CFG, offline=True, now=kw.pop("now", SUN_9AM), **kw), sent


def test_pipeline_end_to_end():
    _reset_outputs()
    res, sent = _run()
    assert res["status"] == "sent" and sent and res["buys"]
    for f in ("signals/latest.json", "signals/table.csv", "signals/regime.csv", "signals/coverage.csv", "signals/history.csv",
              "signals/universe.csv", "report/latest.json", "backtest/summary.json"):
        assert (TMP / f).exists(), f
    # A dated universe snapshot is kept for point-in-time audits and to track
    # future EGX33 / external-screen membership changes without overwriting history.
    snap = TMP / "universe_history" / "2026-09-20.csv"
    assert snap.exists(), "dated Sharia-universe snapshot missing"
    snap_df = pd.read_csv(snap)
    assert "snapshot_date" in snap_df.columns and "sharia_verified" in snap_df.columns
    latest_payload = json.loads((C.SIGNALS / "latest.json").read_text(encoding="utf-8"))
    coverage = latest_payload.get("history_coverage", {})
    assert coverage.get("requested_years") == float(CFG["data"]["history_years"])
    assert "median_years" in coverage and "symbols_at_least_5y" in coverage
    assert latest_payload.get("universe", {}).get("sharia_verified_total", 0) >= 0
    assert C.load_state()["last_sent_date"] == "2026-09-20"
    # نفس اليوم مرة ثانية = تخطي، ولا رسالة جديدة
    n = len(sent)
    res2, sent2 = _run()
    assert res2["status"] == "skipped" and not sent2
    # الإجازة الأسبوعية
    assert _run(now=datetime(2026, 9, 18, 9, 5, tzinfo=CAIRO))[0]["status"] == "skipped"
    # الاستبعاد ظاهر في الجدول
    tbl = pd.read_csv(TMP / "signals" / "table.csv")
    sig = dict(zip(tbl["code"], tbl["signal"]))
    assert sig["ILLQ"] == "ILLIQUID" and sig["STAL"] == "STALE" and "SHRT" not in sig


def test_pipeline_dry_run_and_missing_secrets_and_alerts():
    _reset_outputs()
    res, sent = _run(dry_run=True)
    assert res["status"] == "dry_run" and not sent and "last_sent_date" not in C.load_state()

    _reset_outputs()
    pipeline.send_message = lambda *a, **k: 1
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    assert pipeline.run(CFG, offline=True, now=SUN_9AM)["status"] == "not_sent"

    # بيانات ناقصة = خطأ + تنبيه على تلجرام
    _reset_outputs()
    alerts = []
    os.environ["TELEGRAM_BOT_TOKEN"] = "t"
    pipeline.send_message = lambda token, chat, text: (alerts.append(text) or 1)
    orig = pipeline.load_prices
    pipeline.load_prices = lambda codes, cfg, now: {}
    try:
        pipeline.run(CFG, offline=True, now=SUN_9AM)
        raise AssertionError("كان لازم يرفع DataError")
    except pipeline.DataError:
        pass
    finally:
        pipeline.load_prices = orig
    assert alerts and "تعذّر" in alerts[0]


def test_backtest_underperformance_is_reported():
    weak = {
        "period": {"from": "2021-10-14", "to": "2026-10-14"},
        "strategy": {"avg_r": 0.109, "profit_factor": 1.20},
        "baseline": {"avg_r": 0.165, "profit_factor": 1.31},
    }
    warning = performance_warning(weak)
    assert warning and "لا يتفوق على خط الأساس" in warning
    assert "2021-10-14" in warning and "1.20" in warning
    assert performance_warning({
        "strategy": {"avg_r": 0.20, "profit_factor": 1.4},
        "baseline": {"avg_r": 0.10, "profit_factor": 1.2},
    }) is None
    assert performance_warning({}) is None


def test_unverified_candidates_do_not_consume_buy_slots():
    frames, regime_df, _, _ = _analysis()
    names = {c: c for c in frames}
    latest = latest_table(frames, names, CFG)
    candidates = latest[latest["buy_raw"] & ~latest["stale"] & latest["liquid"]].sort_values(["score", "rs_pct"], ascending=False)
    if len(candidates) < 2:
        return
    latest["sharia_verified"] = True
    # Highest-scoring candidate is unverified; the next eligible one must retain the slot.
    latest.loc[candidates.index[0], "sharia_verified"] = False
    last = regime_df.iloc[-1]
    cls = classify(latest, last["regime"], last["regime_score"], CFG)
    assert not ((cls["signal"] == "BUY") & ~cls["sharia_verified"]).any()
    if int(CFG["risk"]["max_buys_by_regime"][last["regime"]]) > 0:
        verified_eligible = latest.loc[latest["sharia_verified"] & latest["buy_raw"] & ~latest["stale"] & latest["liquid"]]
        assert len(cls[cls["signal"] == "BUY"]) == min(len(verified_eligible), int(CFG["risk"]["max_buys_by_regime"][last["regime"]]))


def test_price_clean_rejects_impossible_ohlc_candles():
    from src.data_fetch import _clean

    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    frame = pd.DataFrame({
        "open": [10.0, 10.0, 10.0],
        "high": [11.0, 9.0, 11.0],
        "low": [9.0, 8.0, 9.0],
        "close": [10.5, 10.5, 10.0],
        "volume": [100.0, 100.0, -1.0],
    }, index=idx)
    clean = _clean(frame)
    assert list(clean.index) == [idx[0]], "invalid high/low or negative volume must be excluded"


def test_sharia_universe_prefers_egx33_shariah_benchmark_cache():
    from src import data_fetch

    code = "_INDEX_EGX33_SHARIAH"
    path = data_fetch.cache_path(code)
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    try:
        idx = pd.date_range("2022-01-03", periods=80, freq="B")
        close = np.linspace(100, 130, len(idx))
        frame = pd.DataFrame({
            "open": close, "high": close + 2.0, "low": close - 1.0, "close": close,
            "volume": np.full(len(idx), 1000.0),
        }, index=idx)
        data_fetch.write_cache(code, frame)
        cfg = {**CFG, "universe": {**CFG["universe"], "mode": "sharia_union"}}
        result = data_fetch.fetch_benchmark(cfg, {}, offline=True)
        assert result is not None
        assert result.index.min() == idx.min()
        assert np.isclose(result["close"].iloc[-1], frame["close"].iloc[-1], atol=1e-4)
    finally:
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)

def test_history_backfill_is_throttled_for_free_sources():
    from src import data_fetch

    path = data_fetch.cache_path("THROTTLE")
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    original_fetch = data_fetch.fetch_yahoo_from
    calls = []
    try:
        old_idx = pd.date_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3),
            periods=1210, freq="B",
        )
        cached = pd.DataFrame({
            "open": np.full(len(old_idx), 10.0), "high": np.full(len(old_idx), 11.0),
            "low": np.full(len(old_idx), 9.0), "close": np.full(len(old_idx), 10.0),
            "volume": np.full(len(old_idx), 1000.0),
        }, index=old_idx)
        data_fetch.write_cache("THROTTLE", cached)

        def fake_fetch(symbol, start, retries=3):
            calls.append(pd.Timestamp(start).date())
            idx = pd.date_range(
                end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
                periods=3, freq="B",
            )
            return pd.DataFrame({
                "open": np.full(len(idx), 10.0), "high": np.full(len(idx), 11.0),
                "low": np.full(len(idx), 9.0), "close": np.full(len(idx), 10.0),
                "volume": np.full(len(idx), 1000.0),
            }, index=idx)

        data_fetch.fetch_yahoo_from = fake_fetch
        cfg = {**CFG, "data": {**CFG["data"], "history_years": 8, "history_backfill_retry_days": 30, "request_sleep": 0}}
        state = {}
        data_fetch.refresh_all(cfg, ["THROTTLE"], state, offline=False)
        data_fetch.refresh_all(cfg, ["THROTTLE"], state, offline=False)
        assert len(calls) == 2
        assert calls[1] > calls[0], "second run should refresh recent bars, not redownload full history"
        assert state["history_backfill_attempted"]["THROTTLE"]
    finally:
        data_fetch.fetch_yahoo_from = original_fetch
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)


def test_portfolio_requires_explicit_current_sharia_verification_to_increase():
    from src.portfolio import recommend_stock

    base = {"score": 90, "signal": "BUY", "close": 100, "stop": 90}
    # False, missing, and null are all unknown/unapproved; none may increase.
    for status in (False, None, "pending"):
        row = {**base, "sharia_verified": status}
        action, reason = recommend_stock(row, "up", avg_cost=95)
        assert action == "HOLD"
        assert "مراجعة شرعية" in reason
    action, reason = recommend_stock(base, "up", avg_cost=95)
    assert action == "HOLD" and "مراجعة شرعية" in reason

    # An explicit, current verification preserves the normal decision path.
    action, _ = recommend_stock({**base, "sharia_verified": True}, "up", avg_cost=95)
    assert action == "INCREASE"



# ------------------------------------------------------------------ التعلم الآلي: تسميات صحيحة واختبار زمني
def test_ml_dataset_uses_forward_returns_and_drops_unlabelled_tail():
    dates = pd.bdate_range("2024-01-01", periods=40)
    close = pd.Series(np.linspace(10.0, 20.0, len(dates)), index=dates)
    frame = pd.DataFrame({"close": close}, index=dates)
    for col in (
        "score", "rsi14", "macd_hist", "atr_pct", "ret1", "ret5", "ret20",
        "ret60", "vol_ratio", "updown_vol", "rs_rank", "dist_high252", "sma50_slope",
    ):
        frame[col] = 1.0
    horizon = 5
    ds = _dataset({"TEST": frame}, horizon=horizon, threshold=0.01)
    assert len(ds) == len(frame) - horizon
    assert ds["date"].max() == dates[-horizon - 1]
    assert (ds["code"] == "TEST").all()
    expected = (close.shift(-horizon) / close - 1.0 >= 0.01).iloc[:-horizon].to_numpy()
    assert np.array_equal(ds["target"].astype(bool).to_numpy(), expected)
    # The final horizon has no known future outcome and must never be labelled negative.
    assert not (ds["date"] >= dates[-horizon]).any()


def test_ml_fails_closed_when_history_is_too_short():
    dates = pd.bdate_range("2025-01-01", periods=30)
    frame = pd.DataFrame(index=dates)
    frame["close"] = np.linspace(10.0, 12.0, len(dates))
    for col in (
        "score", "rsi14", "macd_hist", "atr_pct", "ret1", "ret5", "ret20",
        "ret60", "vol_ratio", "updown_vol", "rs_rank", "dist_high252", "sma50_slope",
    ):
        frame[col] = 1.0
    probs, meta = train_predict(
        {"TEST": frame},
        {**CFG, "ml": {**CFG["ml"], "enabled": True, "min_samples": 500}},
    )
    assert probs == {}
    assert meta["enabled"] is True
    assert meta.get("model_used") is not True
    assert meta.get("error") == "عينات غير كافية"

# ------------------------------------------------------------------ القائمة
def test_universe_modes_on_real_files():
    real = ROOT / "data"
    lst = pd.read_csv(real / "universe.csv")
    assert len(lst) >= 140 and lst["code"].is_unique
    backup = {n: (TMP / n).read_bytes() for n in ("universe.csv", "egx33.txt")}
    extra_path = TMP / "sharia_extra.csv"
    extra_backup = extra_path.read_bytes() if extra_path.exists() else None
    try:
        shutil.copy(real / "universe.csv", TMP / "universe.csv")
        shutil.copy(real / "egx33.txt", TMP / "egx33.txt")
        shutil.copy(real / "sharia_extra.csv", extra_path)
        cfg = {**CFG, "universe": {"mode": "union"}}
        u = build_universe(cfg)
        both = build_universe({**CFG, "universe": {"mode": "intersection"}})
        only_l = build_universe({**CFG, "universe": {"mode": "list_only"}})
        sharia = build_universe({**CFG, "universe": {"mode": "sharia_union"}})
        assert len(only_l) == len(lst) and len(both) > 0 and len(u) >= len(lst)
        assert set(both["code"]) <= set(only_l["code"]) and set(only_l["code"]) <= set(u["code"])
        assert (u["in_list"] | u["in_egx33"]).all()
        # The personal Sharia universe excludes generic research candidates;
        # only EGX33 constituents and recent, audited external names are included.
        assert len(sharia) < len(lst)
        assert sharia.loc[sharia["in_egx33"], "sharia_verified"].all()
        assert sharia["sharia_verified"].all()
        assert (sharia["in_egx33"] | (sharia["in_sharia_extra"] & sharia["sharia_review_status"].eq("اعتماد خارجي حديث"))).all()
        assert len(sharia.loc[sharia["in_sharia_extra"]]) == 73
    finally:
        for n, b in backup.items():
            (TMP / n).write_bytes(b)
        if extra_backup is None:
            extra_path.unlink(missing_ok=True)
        else:
            extra_path.write_bytes(extra_backup)


def test_sharia_extra_requires_audited_approval():
    path = TMP / "sharia_extra.csv"
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    try:
        pd.DataFrame([
            {"code": "OUT1", "name_ar": "سهم خارجي معتمد", "verification_source": "مصدر شرعي موثق", "verification_url": "https://example.com/sharia-screening",
             "verified_on": "2026-10-01", "notes": "test", "status": "approved"},
            {"code": "OUT2", "name_ar": "سهم غير معتمد", "verification_source": "",
             "verified_on": "", "notes": "test", "status": "pending"},
        ]).to_csv(path, index=False)
        union = build_universe({**CFG, "universe": {"mode": "union"}})
        sharia = build_universe({**CFG, "universe": {"mode": "sharia_union"}})
        assert "OUT1" not in set(union["code"])
        assert "OUT1" in set(sharia["code"])
        # Incomplete candidates remain visible for research, but never qualify as Sharia-verified.
        assert "OUT2" in set(sharia["code"])
        out2 = sharia.loc[sharia["code"] == "OUT2"].iloc[0]
        assert not bool(out2["sharia_verified"])
        assert out2["sharia_review_status"] == "اعتماد خارجي منتهي الصلاحية"
        assert bool(sharia.loc[sharia["code"] == "OUT1", "sharia_verified"].iloc[0])
        stale = pd.DataFrame([{"code":"OUT3","name_ar":"سهم انتهت مراجعته","verification_source":"مصدر موثق","verified_on":"2025-01-01","notes":"","status":"approved"}])
        stale.to_csv(path,index=False)
        expired = build_universe({**CFG,"universe":{"mode":"sharia_union","verification_max_age_days":180}})
        assert "OUT3" in set(expired["code"])
        stale_row = expired.loc[expired["code"] == "OUT3"].iloc[0]
        assert not bool(stale_row["sharia_verified"])
        assert stale_row["sharia_review_status"] == "اعتماد خارجي منتهي الصلاحية"
    finally:
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)



# ------------------------------------------------------------------ الواجهة
def test_ui_html_builders():
    from app import ui

    pick = {"code": "ABCD", "name": "<b>شركة</b> & أخرى", "score": 82.4, "setup_ar": "استمرار اتجاه", "close": 22.5, "entry_low": 22.2,
            "entry_high": 22.7, "stop": 21.8, "t1": 24.0, "t2": 24.8, "avg_value_m": 30.0, "rsi": 58, "rs_pct": 80, "position_pct": 10,
            "why": ["سبب"], "cautions": ["تحذير"]}
    h = ui.pick_html(pick, "تعليق")
    assert "&lt;b&gt;" in h and "<b>شركة" not in h and "21.80" in h and "24.80" in h and "lad-entry" in h
    assert ui.ladder_html({**pick, "stop": 30}) == ""                # وقف فوق السعر = مفيش سلم
    assert ui.ladder_html({"code": "X"}) == ""
    _, _, _, result = _analysis()
    v = ui.verdict_html(result, "ملخص", ["تحذير بيانات"])
    assert result["regime"]["label_ar"] in v and "تحذير بيانات" in v
    assert ui.change_html(float("nan")) == "-" and "▲" in ui.change_html(1.2) and "▼" in ui.change_html(-1.2)
    assert "\n\n" not in ui.pick_html(pick, "x")                     # سطر فاضي بيكسر HTML جوه Markdown


class _Dummy:
    def __call__(self, *a, **k):
        return _Dummy()

    def __getattr__(self, name):
        return _Dummy()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def __getitem__(self, key):
        # Supports the minimal pandas-like column access used by the Streamlit smoke test.
        return pd.Series(dtype=object)

    def __setitem__(self, key, value):
        # The smoke test uses an empty dummy frame; assignment only needs to be accepted.
        return None


class StopApp(Exception):
    pass


class _Col(_Dummy):
    """عمود/تبويب وهمي بيرد بقيم افتراضية معقولة على عناصر الإدخال (عشان الصفحة تكمل)."""

    def multiselect(self, label, options, default=None, **k):
        return default or []

    def slider(self, label, lo, hi, value=None, **k):
        return lo if value is None else value

    def selectbox(self, label, options, index=0, **k):
        return options[index]

    def radio(self, label, options, index=0, **k):
        return options[index]

    def number_input(self, label, min_value=0, max_value=None, value=0, **k):
        return value

    def text_input(self, *a, **k):
        return ""


def _fake_streamlit():
    st = types.ModuleType("streamlit")
    d = _Dummy()

    def cache_data(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

    def tabs(labels):
        return [_Col() for _ in labels]

    def columns(spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Col() for _ in range(n)]

    def multiselect(label, options, default=None, **k):
        return default or []

    def slider(label, lo, hi, value=None, **k):
        return lo if value is None else value

    def selectbox(label, options, index=0, **k):
        return options[index]

    def radio(label, options, index=0, **k):
        return options[index]

    def number_input(label, min_value=0, max_value=None, value=0, **k):
        return value

    def stop():
        raise StopApp()

    st.session_state = {}  # Streamlit session state is a mutable mapping in the real app\n    st.cache_data, st.tabs, st.columns, st.multiselect, st.slider = cache_data, tabs, columns, multiselect, slider
    st.selectbox, st.radio, st.number_input, st.stop = selectbox, radio, number_input, stop
    st.text_input = lambda *a, **k: ""
    st.button = lambda *a, **k: False
    st.column_config = d
    st.__getattr__ = lambda name: d      # noqa: E731  (PEP 562)
    for n in ("set_page_config", "markdown", "dataframe", "plotly_chart", "warning", "info", "caption", "download_button", "metric"):
        setattr(st, n, lambda *a, **k: None)
    return st


def _run_app():
    saved = {k: sys.modules.get(k) for k in ("streamlit", "plotly", "plotly.graph_objects", "plotly.subplots")}
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    go.__getattr__ = lambda name: _Dummy()
    sub = types.ModuleType("plotly.subplots")
    sub.make_subplots = lambda *a, **k: _Dummy()
    plotly.graph_objects, plotly.subplots = go, sub
    sys.modules.update({"streamlit": _fake_streamlit(), "plotly": plotly, "plotly.graph_objects": go, "plotly.subplots": sub})
    for m in [k for k in sys.modules if k in ("app.charts", "app.ui", "app.streamlit_app")]:
        del sys.modules[m]
    try:
        try:
            runpy.run_path(str(ROOT / "app" / "streamlit_app.py"), run_name="__main__")
        except StopApp:
            # st.stop() is the expected Streamlit control-flow exit when the
            # stale-report safety gate blocks the page during this smoke test.
            pass
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for m in [k for k in sys.modules if k in ("app.charts", "app.ui")]:
            del sys.modules[m]



def test_fund_catalog_has_published_sharia_and_gold_entries():
    sharia_names = {f["name"] for f in SHARIA_FUNDS}
    gold_names = {f["name"] for f in GOLD_FUNDS}
    assert "بشائر – صندوق البنك الأهلي السادس وبنك البركة" in sharia_names
    assert "هيرميس لأسهم الشريعة" in sharia_names
    assert "الحياة المتوافق مع الشريعة – بنك الكويت الوطني" in sharia_names
    assert "صندوق استثمار بنك فيصل الإسلامي" in sharia_names
    assert "الشكمجية بكرة للذهب" in gold_names
    assert all(name in FUND_SOURCE_MATCHES for name in sharia_names | gold_names)
    # Validate codes against the public SNDUK catalogue, not old guessed labels.
    assert FUND_SOURCE_MATCHES["بشائر – صندوق البنك الأهلي السادس وبنك البركة"][0] == "BSF"
    assert FUND_SOURCE_MATCHES["هيرميس لأسهم الشريعة"][0] == "HSE"
    assert FUND_SOURCE_MATCHES["أمان المتوافق مع الشريعة"][0] == "FCA"
    assert FUND_SOURCE_MATCHES["الحصن – بنك مصر"][0] == "BM4"
    assert FUND_SOURCE_MATCHES["الشكمجية بكرة للذهب"][0] == "SHK"
    assert FUND_SOURCE_MATCHES["الحياة المتوافق مع الشريعة – بنك الكويت الوطني"][0] == "NHE"
    assert FUND_SOURCE_MATCHES["صندوق استثمار بنك فيصل الإسلامي"][0] == "FIF"
    assert FUND_SOURCE_MATCHES["دهب مباشر"][0] == "MGF"
    assert FUND_SOURCE_MATCHES["جولد مصر"][0] == "CGF"
    # The checked-in catalog must stay synchronized with the full code-level catalog.
    catalog = pd.read_csv(ROOT / "data" / "funds.csv", dtype=str).fillna("")
    catalog_names = set(catalog["name_ar"])
    assert catalog_names == sharia_names | gold_names
    assert len(catalog) == len(catalog_names) == 23
    assert set(catalog.loc[catalog["category"] == "sharia_equity", "reference"]) == {"EGX33"}
    assert set(catalog.loc[catalog["category"] == "gold", "reference"]) == {"XAUUSD"}
    assert catalog["source_code"].is_unique

def test_streamlit_app_smoke_run():
    """تشغيل صفحة Streamlit كاملة ببيانات اختبار وهمية لاكتشاف أخطاء منطق الواجهة."""
    _reset_outputs()
    _run(dry_run=True)
    _run_app()







def test_sharia_universe_requires_recent_source_for_out_of_index_shares():
    """Reference universe alone must not imply Sharia eligibility; external approvals expire."""
    import src.universe as universe_module

    old_data = universe_module.DATA
    test_data = TMP / "universe_gate"
    test_data.mkdir(parents=True, exist_ok=True)
    try:
        universe_module.DATA = test_data
        pd.DataFrame([
            {"code": "INLIST", "name_ar": "مرشح مرجعي فقط"},
            {"code": "OUTOLD", "name_ar": "اعتماد قديم"},
            {"code": "OUTNEW", "name_ar": "اعتماد حديث"},
            {"code": "OUTBAD", "name_ar": "بلا مصدر"},
            {"code": "NOURL", "name_ar": "مصدر بلا رابط"},
        ]).to_csv(test_data / "universe.csv", index=False)
        (test_data / "egx33.txt").write_text("INDEX,عضو المؤشر\n", encoding="utf-8")
        pd.DataFrame([
            {"code": "OUTOLD", "name_ar": "اعتماد قديم", "verification_source": "اختبار", "verification_url": "https://example.org/old", "verified_on": "2024-01-01", "notes": "", "status": "approved"},
            {"code": "OUTNEW", "name_ar": "اعتماد حديث", "verification_source": "اختبار", "verification_url": "https://example.org/new", "verified_on": datetime.now().date().isoformat(), "notes": "", "status": "approved"},
            {"code": "OUTBAD", "name_ar": "بلا مصدر", "verification_source": "", "verification_url": "", "verified_on": datetime.now().date().isoformat(), "notes": "", "status": "approved"},
            {"code": "NOURL", "name_ar": "مصدر بلا رابط", "verification_source": "مصدر مسمى فقط", "verification_url": "", "verified_on": datetime.now().date().isoformat(), "notes": "", "status": "approved"},
        ]).to_csv(test_data / "sharia_extra.csv", index=False)
        cfg = {"universe": {"mode": "sharia_union", "verification_max_age_days": 180}}
        result = universe_module.build_universe(cfg).set_index("code")
        assert bool(result.loc["INDEX", "sharia_verified"])
        assert "INLIST" not in result.index  # universe.csv is not proof of compliance
        assert not bool(result.loc["OUTOLD", "sharia_verified"])
        assert bool(result.loc["OUTNEW", "sharia_verified"])
        assert not bool(result.loc["OUTBAD", "sharia_verified"])
        assert not bool(result.loc["NOURL", "sharia_verified"])
    finally:
        universe_module.DATA = old_data


def test_market_freshness_gate_suppresses_buys_and_preserves_fresh_signals():
    from datetime import date
    from src.data_quality import apply_market_freshness_gate

    sample = pd.DataFrame({
        "code": ["A", "B"],
        "signal": ["BUY", "WATCH"],
        "why_not_buy": [[], ["existing rationale"]],
    })
    blocked, did_block = apply_market_freshness_gate(sample, "2026-10-27", date(2026, 10, 28))
    assert did_block
    assert blocked["signal"].tolist() == ["WATCH", "WATCH"]
    assert any("توقف آمن" in x for x in blocked.loc[0, "why_not_buy"])
    assert blocked.loc[1, "why_not_buy"] == ["existing rationale"]
    # A snapshot at or beyond the expected session must remain unchanged.
    fresh, did_block = apply_market_freshness_gate(sample, "2026-10-28", date(2026, 10, 28))
    assert not did_block
    assert fresh is sample
    assert fresh["signal"].tolist() == ["BUY", "WATCH"]


def test_history_lookback_and_walk_forward_ml_are_configured():
    """Eight-year lookback is a request, not a claim that every symbol has eight years."""
    cfg = C.load_config()
    assert int(cfg["data"]["history_years"]) >= 8
    assert cfg["ml"].get("purge_horizon", True)
    assert int(cfg["ml"].get("validation_folds", 0)) >= 2


def test_synthetic_sharia_index_does_not_forward_fill_missing_prices():
    from src.pipeline import build_equal_weight_index

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    # B starts late and A has a missing observation. Neither gap may be turned
    # into an artificial 0% return by pandas' pct_change default behavior.
    a = pd.Series([100.0, 110.0, np.nan, 121.0, 133.1], index=dates, name="A")
    b = pd.Series([np.nan, np.nan, 50.0, 55.0, 60.5], index=dates, name="B")
    result = build_equal_weight_index([a, b], min_constituents=1)
    assert result is not None
    assert result.name == "synthetic_sharia_equal_weight"
    # On the date A is missing, A contributes no return; B contributes its
    # actual 10% return. A forward-filled A would incorrectly dilute this.
    assert np.isclose(result.loc[dates[3]] / result.loc[dates[2]] - 1.0, 0.10)
    # The first output point is the first date with a valid constituent return.
    assert result.index[0] == dates[1]


def test_short_yahoo_backfill_uses_budgeted_eodhd_for_older_history():
    """A successful-but-short Yahoo response must not silently cap the 8-year cache."""
    from src import data_fetch

    code = "BACKFILL"
    path = data_fetch.cache_path(code)
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    old_token = os.environ.get("EODHD_API_TOKEN")
    original_yahoo = data_fetch.fetch_yahoo_from
    original_eodhd = data_fetch.fetch_eodhd
    original_sleep = data_fetch.time.sleep
    calls = []
    today = pd.Timestamp.today().normalize()
    try:
        os.environ["EODHD_API_TOKEN"] = "test-token"
        # Simulate a cache with only five years, despite an eight-year request.
        cached_idx = pd.date_range(end=today - pd.Timedelta(days=4), periods=1200, freq="B")
        cached = pd.DataFrame({
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0,
        }, index=cached_idx)
        data_fetch.write_cache(code, cached)

        def short_yahoo(symbol, start, retries=3):
            idx = pd.date_range(end=today - pd.Timedelta(days=1), periods=4, freq="B")
            return pd.DataFrame({
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0,
            }, index=idx)

        def older_eodhd(symbol, token, start):
            calls.append((symbol, token, pd.Timestamp(start).date()))
            idx = pd.date_range(start=pd.Timestamp(start), end=today - pd.Timedelta(days=5), freq="B")
            return pd.DataFrame({
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0,
            }, index=idx)

        data_fetch.fetch_yahoo_from = short_yahoo
        data_fetch.fetch_eodhd = older_eodhd
        data_fetch.time.sleep = lambda *_args, **_kwargs: None
        cfg = {**CFG, "data": {**CFG["data"], "history_years": 8, "history_backfill_retry_days": 30, "request_sleep": 0, "eodhd_daily_budget": 2}}
        state = {"eodhd": {"date": today.date().isoformat(), "calls": 0}}
        coverage = data_fetch.refresh_all(cfg, [code], state, offline=False)
        assert len(calls) == 1, "EODHD should be used once when Yahoo returns a short backfill"
        assert state["eodhd"]["calls"] == 1, "fallback must count against the daily EODHD budget"
        assert float(coverage.iloc[0]["coverage_years"]) >= 7.7
        assert "eodhd" in str(coverage.iloc[0]["source"]).lower()
    finally:
        data_fetch.fetch_yahoo_from = original_yahoo
        data_fetch.fetch_eodhd = original_eodhd
        data_fetch.time.sleep = original_sleep
        if old_token is None:
            os.environ.pop("EODHD_API_TOKEN", None)
        else:
            os.environ["EODHD_API_TOKEN"] = old_token
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)



def test_personal_sharia_universe_includes_index_and_audited_extras_only():
    """EGX33 members + fresh audited additions; generic watchlist is not a Sharia pass."""
    from datetime import date
    from src import universe as U

    U.DATA.mkdir(parents=True, exist_ok=True)
    paths = ("universe.csv", "egx33.txt", "sharia_extra.csv")
    backups = {name: (U.DATA / name).read_bytes() if (U.DATA / name).exists() else None for name in paths}
    try:
        (U.DATA / "universe.csv").write_text(
            "code,name_ar\nGENERIC,مرشح عام\nIN33,اسم المؤشر\n",
            encoding="utf-8",
        )
        (U.DATA / "egx33.txt").write_text(
            "# synthetic fixture\nIN33,Index constituent\n",
            encoding="utf-8",
        )
        today = date.today().isoformat()
        (U.DATA / "sharia_extra.csv").write_text(
            "code,name_ar,verification_source,verification_url,verified_on,notes,status\n"
            f"EXTRA,إضافة موثقة,Test screening,https://example.org/screen.pdf,{today},fixture,approved\n"
            f"STALE,اعتماد قديم,Test screening,https://example.org/old.pdf,2020-01-01,fixture,approved\n"
            f"REJECT,مرفوض,Test screening,https://example.org/reject.pdf,{today},fixture,rejected\n"
            f"BADURL,رابط غير صالح,Test screening,not-a-url,{today},fixture,approved\n",
            encoding="utf-8",
        )
        out = U.build_universe({"universe": {"mode": "sharia_union", "verification_max_age_days": 180}})
        by_code = out.set_index("code")
        assert set(by_code.index) == {"IN33", "EXTRA", "STALE", "REJECT", "BADURL"}
        assert bool(by_code.loc["IN33", "sharia_verified"])
        assert bool(by_code.loc["EXTRA", "sharia_verified"])
        assert not bool(by_code.loc["STALE", "sharia_verified"])
        assert not bool(by_code.loc["REJECT", "sharia_verified"])
        assert not bool(by_code.loc["BADURL", "sharia_verified"])
        assert "GENERIC" not in by_code.index
    finally:
        for name, backup in backups.items():
            path = U.DATA / name
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(backup)



def test_report_attempt_guard_prevents_duplicate_daily_send():
    from src.pipeline import should_run
    from src import config as C
    cfg = C.load_config()
    now = datetime(2026, 10, 28, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
    state = {"report_attempt_date": now.date().isoformat()}
    allowed, reason = should_run(cfg, now, state)
    assert not allowed
    assert "محاولة إرسال" in reason

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
