"""اختبار القواعد على التاريخ. نفس قواعد التوصية الحالية ونفس محاكاة الصفقة.

تحذيرات مهمة (تظهر أيضًا في التطبيق):
- القائمة الحالية للأسهم = تحيّز البقاء (الأسهم اللي اختفت أو اتوقفت مش موجودة).
- لا يوجد نموذج انزلاق خاص بسيولة كل سهم؛ يعرض التقرير اختبارات ضغط باحتكاك إضافي لتوضيح حساسية النتائج.
- النتائج السابقة لا تضمن المستقبل.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sim import simulate_trade


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["r_net"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else None
    # أطول سلسلة خسائر متتالية
    streak = best = 0
    for x in trades.sort_values("exit_date")["r_net"]:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    eq = trades.sort_values("exit_date")["r_net"].cumsum()
    dd = float((eq - eq.cummax()).min())
    return {
        "trades": int(len(trades)),
        "win_rate": round(float((r > 0).mean() * 100), 1),
        "avg_r": round(float(r.mean()), 3),
        "median_r": round(float(np.median(r)), 3),
        "profit_factor": None if pf is None else round(pf, 2),
        "avg_ret_pct": round(float(trades["ret_pct"].mean()), 2),
        "avg_days": round(float(trades["days"].mean()), 1),
        "max_losing_streak": int(best),
        "max_drawdown_r": round(dd, 2),
        "pct_target": round(float((trades["reason"] == "target").mean() * 100), 1),
        "pct_stop": round(float((trades["reason"] == "stop").mean() * 100), 1),
        "pct_time": round(float((trades["reason"] == "time").mean() * 100), 1),
    }


def _run_symbol(code: str, f: pd.DataFrame, mask: np.ndarray, cfg: dict, step: int = 1) -> list[dict]:
    o, h, l, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    stop, t1 = f["stop"].to_numpy(dtype=float), f["t1"].to_numpy(dtype=float)
    dates = f.index
    rk = cfg["risk"]
    out = []
    i, n = 0, len(f)
    while i < n - 1:
        if mask[i] and np.isfinite(stop[i]):
            res = simulate_trade(o, h, l, c, i, stop[i], t1[i], rk["max_hold_days"], rk["round_trip_cost_pct"])
            if res["status"] == "closed":
                out.append(
                    {
                        "code": code,
                        "signal_date": dates[i].date().isoformat(),
                        "entry_date": dates[res["entry_idx"]].date().isoformat(),
                        "exit_date": dates[res["exit_idx"]].date().isoformat(),
                        "entry": res["entry"],
                        "exit": res["exit"],
                        "risk": float(res["entry"] - stop[i]),
                        "reason": res["reason"],
                        "r_net": res["r_net"],
                        "ret_pct": res["ret_pct"],
                        "days": res["days"],
                        "score": float(f["score"].iloc[i]),
                        "value_sma20": float(f["value_sma20"].iloc[i]) if "value_sma20" in f.columns else None,
                    }
                )
                i = res["exit_idx"] + 1      # لا نفتح صفقة جديدة في نفس السهم قبل خروج القديمة
                continue
            # skipped/open/pending: نكمل
        i += step
    return out


def _collect_trades(frames: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both strategy and baseline under the configured round-trip friction."""
    strat, base = [], []
    step = cfg["backtest"]["baseline_every_n_bars"]
    for code, f in frames.items():
        if len(f) < 260:
            continue
        valid = f["stop"].notna() & f["liquid"]
        strat += _run_symbol(code, f, (f["buy_raw"] & f["regime_ok"]).to_numpy(), cfg)
        base += _run_symbol(code, f, valid.to_numpy(), cfg, step=step)
    return pd.DataFrame(strat), pd.DataFrame(base)


def _add_extra_friction(trades: pd.DataFrame, extra_pct) -> pd.DataFrame:
    """تطبيق احتكاك إضافي ذهابًا وإيابًا دون إعادة محاكاة المخارج.

    يقبل رقمًا ثابتًا أو متغيّرًا لكل صفقة (مثل انزلاق السيولة).
    """
    out = trades.copy()
    if out.empty:
        return out
    if isinstance(extra_pct, pd.Series):
        extra = pd.Series(extra_pct, index=out.index).astype(float).fillna(0.0).to_numpy()
    else:
        extra = np.full(len(out), float(extra_pct))
    extra_cost = (extra / 100.0) * out["entry"].astype(float).to_numpy()
    risk = out["risk"].astype(float).replace(0, np.nan).to_numpy()
    out["r_net"] = out["r_net"].astype(float) - extra_cost / risk
    out["ret_pct"] = out["ret_pct"].astype(float) - extra
    return out


def liquidity_slippage_pct(trades: pd.DataFrame, cfg: dict) -> pd.Series:
    """تقدير انزلاق سعري لكل صفقة حسب سيولة السهم بدل نسبة ثابتة للجميع.

    تقريب جذر الأمر (square-root impact): ينمو الانزلاق مع الجذر التربيعي
    لنسبة قيمة الأمر إلى متوسط قيمة التداول اليومي للسهم، مع سقف يحمي من
    قيم شاذة في الأسهم الرقيقة. تقدير مبسط وليس نموذج تنفيذ دقيقًا.
    """
    out = pd.Series(0.0, index=trades.index, dtype=float)
    ls = cfg.get("backtest", {}).get("liquidity_slippage", {})
    if trades.empty or not ls.get("enabled", False):
        return out
    order_value = max(1.0, float(ls.get("order_value_egp", 100_000)))
    coef = float(ls.get("coefficient", 0.005))
    cap = float(ls.get("cap_pct", 1.0))
    if "value_sma20" not in trades.columns:
        return out
    adv = pd.to_numeric(trades["value_sma20"], errors="coerce").astype(float)
    adv = adv.where(adv > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        slip = coef * np.sqrt(order_value / adv) * 100.0
    slip = slip.clip(upper=cap).fillna(0.0)
    return slip.reset_index(drop=True).set_axis(trades.index)


def run_backtest(frames: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """يرجع (ملخص، صفقات الاستراتيجية، صفقات خط الأساس).

    يعرض أيضًا حساسية الأداء لاحتكاك تداول إضافي. هذه سيناريوهات ضغط للتكلفة،
    وليست نموذج انزلاق سعري خاصًا بكل سهم أو تقديرًا دقيقًا لتأثير السوق.
    """
    base_cost = float(cfg["risk"]["round_trip_cost_pct"])
    st, bs = _collect_trades(frames, cfg)
    summary = {
        "strategy": _stats(st),
        "baseline": _stats(bs),
        "period": {
            "from": min(f.index[0] for f in frames.values()).date().isoformat(),
            "to": max(f.index[-1] for f in frames.values()).date().isoformat(),
        },
        "assumptions": {
            "entry": "افتتاح الجلسة التالية",
            "max_hold_days": cfg["risk"]["max_hold_days"],
            "round_trip_cost_pct": base_cost,
            "target_r": cfg["risk"]["target_r"][0],
            "extra_friction_scenarios_pct": [0.25, 0.50],
            "friction_note": (
                "اختبارات ضغط باحتكاك إضافي على الصفقة ذهابًا وإيابًا؛ "
                "ليست محاكاة انزلاق تعتمد على سيولة كل سهم أو حجم الأمر."
            ),
        },
        "cost_sensitivity": {
            "base_round_trip_cost_pct": base_cost,
            "scenarios": [],
        },
        # Current universe is not a historical point-in-time membership database.
        # Surface this caveat alongside metrics so historical results aren't
        # mistaken for a survivorship-bias-free simulation.
        "universe_scope": {
            "mode": cfg.get("universe", {}).get("mode", "unspecified"),
            "membership_basis": "current_universe_applied_to_available_history",
            "point_in_time_membership": False,
            "survivorship_bias_warning": (
                "الاختبار يطبق قائمة الأسهم الحالية على التاريخ المتاح؛ "
                "لا يشمل تلقائيًا الشركات المشطوبة أو تغيرات التوافق/العضوية التاريخية. "
                "النتائج استكشافية وليست اختبارًا تاريخيًا خاليًا من تحيز البقاء."
            ),
        },
    }
    for extra in (0.25, 0.50):
        stress_strategy = _add_extra_friction(st, extra)
        stress_baseline = _add_extra_friction(bs, extra)
        summary["cost_sensitivity"]["scenarios"].append({
            "additional_friction_pct": extra,
            "total_round_trip_cost_pct": round(base_cost + extra, 4),
            "strategy": _stats(stress_strategy),
            "baseline": _stats(stress_baseline),
        })

    # انزلاق سيولة لكل سهم: يُقدَّر من متوسط قيمة التداول اليومي وقت الإشارة
    # ويُضاف فوق التكلفة الأساسية كسيناريو تنفيذ أكثر واقعية للأسهم الرقيقة.
    liq = cfg.get("backtest", {}).get("liquidity_slippage", {})
    if liq.get("enabled", False) and not st.empty:
        slip = liquidity_slippage_pct(st, cfg)
        st_liq = _add_extra_friction(st, slip)
        liq_section = {
            "enabled": True,
            "order_value_egp": liq.get("order_value_egp", 100_000),
            "coefficient": liq.get("coefficient", 0.005),
            "cap_pct": liq.get("cap_pct", 1.0),
            "median_slip_pct": round(float(slip.median()), 3),
            "max_slip_pct": round(float(slip.max()), 3),
            "strategy": _stats(st_liq),
        }
        if not bs.empty:
            bs_liq = _add_extra_friction(bs, liquidity_slippage_pct(bs, cfg))
            liq_section["baseline"] = _stats(bs_liq)
        summary["liquidity_slippage"] = liq_section
    return summary, st, bs


def performance_warning(summary: dict) -> str | None:
    """Warn when the current rule set fails to beat its baseline on core metrics."""
    strategy = summary.get("strategy") or {}
    baseline = summary.get("baseline") or {}
    if not strategy or not baseline:
        return None
    try:
        strategy_avg = float(strategy["avg_r"])
        baseline_avg = float(baseline["avg_r"])
        strategy_pf = float(strategy["profit_factor"])
        baseline_pf = float(baseline["profit_factor"])
    except (KeyError, TypeError, ValueError):
        return None
    if strategy_avg < baseline_avg and strategy_pf < baseline_pf:
        period = summary.get("period") or {}
        start, end = period.get("from"), period.get("to")
        period_text = f" خلال {start} إلى {end}" if start and end else ""
        return (
            "مراجعة أداء الاستراتيجية: الاختبار التاريخي الحالي"
            f"{period_text} لا يتفوق على خط الأساس في متوسط R "
            f"({strategy_avg:.3f} مقابل {baseline_avg:.3f}) أو عامل الربح "
            f"({strategy_pf:.2f} مقابل {baseline_pf:.2f}). "
            "لا تعتبر كثرة الإشارات دليلًا على الجودة؛ راجع نتائج الاختبار وتحيز البقاء "
            "والتكاليف قبل الاعتماد على الاستراتيجية."
        )
    return None

