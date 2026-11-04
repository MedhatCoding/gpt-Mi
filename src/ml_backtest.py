"""اختبار تاريخي (walk-forward) لطبقة التعلم الآلي فوق إشارات النظام التقنية.

المنهجية: نقسم التاريخ إلى نوافذ زمنية متتالية؛ قبل كل نافذة يُدرَّب
النموذج على البيانات السابقة فقط (مع فاصل purge بحجم أفق التنبؤ لمنع
تسريب التسميات)، ثم نحسب احتمالات النجاح لصفقات النافذة ونقارن
"التقني فقط" مع "التقني + فلتر احتمال النموذج" داخل النوافذ نفسها.
النتيجة تقيس هل فلتر التعلم الآلي يحسّن جودة الصفقات فعلاً أم لا.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .backtest import _run_symbol, _stats
from .ml_model import FEATURES, _dataset


def _make_models(mcfg: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rf = RandomForestClassifier(
        n_estimators=int(mcfg.get("rf_trees", 120)),
        max_depth=int(mcfg.get("rf_depth", 6)),
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    deep = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=0.0005, batch_size=128, learning_rate_init=0.001,
            max_iter=int(mcfg.get("mlp_max_iter", 40)),
            early_stopping=False, random_state=seed,
        )),
    ])
    return rf, deep


def run_ml_overlay_backtest(frames: dict[str, pd.DataFrame], cfg: dict, eligible_codes: set[str] | None = None) -> dict:
    mcfg = dict(cfg.get("ml", {}))
    bcfg = dict(mcfg.get("backtest", {}))
    if not mcfg.get("enabled", True) or not bcfg.get("enabled", False):
        return {"enabled": False, "note": "اختبار التعلم الآلي التاريخي معطل في الإعدادات"}

    # نفس نطاق تدريب الإنتاج: الأسهم المؤهلة شرعيًا فقط لو مررت.
    ml_frames = frames if eligible_codes is None else {k: v for k, v in frames.items() if k in eligible_codes}

    horizon = int(mcfg.get("horizon_days", 10))
    net_threshold = float(mcfg.get("positive_return", 0.03))
    cost_pct = float(cfg.get("risk", {}).get("round_trip_cost_pct", 0.0))
    gross_threshold = net_threshold + max(0.0, cost_pct) / 100.0
    data = _dataset(ml_frames, horizon, gross_threshold)
    min_samples = int(mcfg.get("min_samples", 500))
    if len(data) < min_samples:
        return {"enabled": True, "samples": int(len(data)),
                "error": "عينات غير كافية للاختبار التاريخي للنموذج"}

    window_days = max(20, int(bcfg.get("window_days", 60)))
    max_windows = max(1, int(bcfg.get("max_windows", 5)))
    prob_min = float(bcfg.get("prob_min", 0.55))
    min_train_days = int(bcfg.get("min_train_days", 400))
    max_train_rows = max(5000, int(bcfg.get("max_train_rows", 100_000)))
    mlp_iters = int(bcfg.get("mlp_max_iter", 40))

    dates = pd.to_datetime(data["date"])
    unique_dates = pd.Index(dates.drop_duplicates().sort_values())
    n = len(unique_dates)
    cols = [c for c in FEATURES if c in data.columns]
    X = data[cols].astype(float)
    y = data["target"].astype(int)

    prob_map: dict[tuple[str, str], float] = {}
    windows_meta = []
    y_true: list[int] = []
    y_prob: list[float] = []
    used = 0
    for k in range(max_windows, 0, -1):
        start_idx = n - k * window_days
        end_idx = min(n, start_idx + window_days)
        train_end_idx = start_idx - horizon
        if start_idx <= max(20, min_train_days) or train_end_idx < 20 or end_idx - start_idx < 10:
            continue
        win_start, win_end = unique_dates[start_idx], unique_dates[end_idx - 1]
        train_cutoff = unique_dates[train_end_idx]
        train_mask = dates < train_cutoff
        win_mask = (dates >= win_start) & (dates <= win_end)
        Xtr, ytr = X.loc[train_mask], y.loc[train_mask]
        Xw = X.loc[win_mask]
        if len(Xtr) < max(50, int(min_samples * 0.35)) or len(Xw) < 10:
            continue
        if ytr.nunique() < 2:
            continue
        # تقليص حجم التدريب للأحدث max_train_rows صفاً للبقاء داخل ميزانية الحوسبة
        if len(Xtr) > max_train_rows:
            Xtr, ytr = Xtr.iloc[-max_train_rows:], ytr.iloc[-max_train_rows:]
        o_cfg = dict(mcfg)
        o_cfg["mlp_max_iter"] = mlp_iters
        rf, deep = _make_models(o_cfg, 42 + used)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf.fit(Xtr, ytr)
            deep.fit(Xtr, ytr)
        pw = 0.55 * rf.predict_proba(Xw)[:, 1] + 0.45 * deep.predict_proba(Xw)[:, 1]
        win_codes = data.loc[win_mask, "code"].astype(str)
        win_dates = dates.loc[win_mask]
        for code, d, pr in zip(win_codes, win_dates, pw):
            prob_map[(code, str(pd.Timestamp(d).date()))] = float(pr)
        y_true.extend(y.loc[win_mask].tolist())
        y_prob.extend(pw.tolist())
        windows_meta.append({
            "train_samples": int(len(Xtr)),
            "window_samples": int(len(Xw)),
            "from": str(pd.Timestamp(win_start).date()),
            "to": str(pd.Timestamp(win_end).date()),
        })
        used += 1

    if used < 2:
        return {"enabled": True, "samples": int(len(data)), "windows": used,
                "error": "تاريخ غير كافٍ لنافذتي اختبار زمني على الأقل"}

    oof_auc = None
    try:
        if len(set(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            oof_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        oof_auc = None

    strat_trades, tech_trades = [], []
    for code, f in ml_frames.items():
        if len(f) < 260:
            continue
        base = (f["buy_raw"] & f["regime_ok"]).to_numpy()
        dkeys = [str(d.date()) for d in f.index]
        prob_arr = np.array([prob_map.get((code, k), 0.0) for k in dkeys])
        in_window = prob_arr > 0.0          # صف بلا احتمال = خارج نوافذ الاختبار
        overlay = base & (prob_arr >= prob_min)
        tech = base & in_window
        strat_trades += _run_symbol(code, f, overlay, cfg)
        tech_trades += _run_symbol(code, f, tech, cfg)

    st_df, tech_df = pd.DataFrame(strat_trades), pd.DataFrame(tech_trades)
    kept = (len(st_df) / len(tech_df) * 100.0) if len(tech_df) else None
    return {
        "enabled": True,
        "samples": int(len(data)),
        "windows": used,
        "window_days": window_days,
        "prob_min": prob_min,
        "oof_auc": oof_auc,
        "period": {"from": windows_meta[0]["from"], "to": windows_meta[-1]["to"]},
        "windows_detail": windows_meta,
        "technical": _stats(tech_df),
        "ml_filtered": _stats(st_df),
        "kept_pct": round(kept, 1) if kept is not None else None,
        "assumptions": (
            "النموذج يُدرَّب قبل كل نافذة على بيانات سابقة فقط (walk-forward مع purge) "
            "ويعمل كفلتر احتمال فوق إشارات النظام التقنية داخل النوافذ نفسها؛ "
            "تقريب بتقليص حجم التدريب وليس اختبارًا تاريخيًا كاملًا خاليًا من التحيزات."
        ),
    }


def ml_overlay_warning(summary: dict) -> str | None:
    """تحذير لو فلتر التعلم الآلي لم يُظهر تحسنًا في جودة الصفقات تاريخيًا."""
    if not summary or not summary.get("enabled") or summary.get("error"):
        return None
    tech = summary.get("technical") or {}
    ml = summary.get("ml_filtered") or {}
    if not tech.get("trades") or not ml.get("trades"):
        return None
    try:
        tech_avg = float(tech["avg_r"])
        ml_avg = float(ml["avg_r"])
        tech_pf, ml_pf = tech.get("profit_factor"), ml.get("profit_factor")
    except (KeyError, TypeError, ValueError):
        return None
    if ml_avg < tech_avg and (
        tech_pf is None or ml_pf is None or float(ml_pf) < float(tech_pf)
    ):
        return (
            "مراجعة طبقة التعلم الآلي: الاختبار التاريخي (نوافذ walk-forward) لم يُظهر تحسنًا في جودة "
            f"الصفقات بعد فلترة الاحتمال (متوسط R {ml_avg:.3f} مقابل {tech_avg:.3f} للتقني فقط، "
            f"وعامل الربح {ml_pf} مقابل {tech_pf}). الفلتر ما زال يعمل، لكن راجع حد الاحتمال والخصائص "
            "قبل الاعتماد عليه."
        )
    return None
