"""تعلم آلي + شبكة عصبية متعددة الطبقات للتنبؤ الاحتمالي بفرص 10 جلسات قادمة.
الهدف ليس استبدال التحليل الفني، بل إضافة طبقة تعلم من سجل الأسعار التاريخي.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURES = [
    "score", "rsi14", "macd_hist", "atr_pct", "ret1", "ret5", "ret20", "ret60",
    "vol_ratio", "updown_vol", "rs_rank", "dist_high252", "sma50_slope",
    "ret120", "dist_sma200", "breakout_dist",
]


def _fit_platt(p, y):
    """Platt scaling: انحدار لوجستي فوق درجات المصنف لتحويلها إلى احتمال معاير.

    يُدرَّب على توقعات خارج العينة (OOF) ويُطبَّق على درجات الإنتاج فقط إذا
    حسَّن Brier مقارنة بالدرجات الخام؛ يرجع None لو الفئة الواحدة أو الفشل.
    """
    try:
        arr = np.asarray(y)
        if len(np.unique(arr)) < 2:
            return None
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=200)
        lr.fit(np.asarray(p, dtype=float).reshape(-1, 1), arr)
        return lr
    except Exception:
        return None

def _dataset(frames: dict[str, pd.DataFrame], horizon: int, threshold: float) -> pd.DataFrame:
    rows = []
    for code, f in frames.items():
        x = f.copy()
        cols = [c for c in FEATURES if c in x.columns]
        if len(cols) < 8:
            continue
        future = x["close"].shift(-horizon) / x["close"] - 1.0
        z = x[cols].replace([np.inf, -np.inf], np.nan).copy()
        z["target"] = (future >= threshold).astype(float)
        z["code"] = code
        z["date"] = x.index
        z = z.loc[future.notna()].dropna()
        if len(z):
            rows.append(z)
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

def train_predict(frames: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, float], dict]:
    mcfg = cfg.get("ml", {})
    if not mcfg.get("enabled", True):
        return {}, {"enabled": False, "samples": 0, "auc": None}

    horizon = int(mcfg.get("horizon_days", 10))
    # Target is defined on net return: require the desired net gain PLUS round-trip
    # trading costs, otherwise the model would learn "winners" that may be flat
    # or negative after commissions and slippage.
    net_threshold = float(mcfg.get("positive_return", 0.03))
    cost_pct = float(cfg.get("risk", {}).get("round_trip_cost_pct", 0.0))
    cost_fraction = max(0.0, cost_pct) / 100.0
    gross_threshold = net_threshold + cost_fraction
    data = _dataset(frames, horizon, gross_threshold)
    if len(data) < int(mcfg.get("min_samples", 500)):
        return {}, {"enabled": True, "samples": int(len(data)), "auc": None, "error": "عينات غير كافية"}

    cols = [c for c in FEATURES if c in data.columns]
    X = data[cols].astype(float)
    y = data["target"].astype(int)

    # Walk-forward validation: evaluate several consecutive unseen periods.
    # Split by session date (all symbols from a date stay together) and purge the
    # forward-return horizon before each validation window to reduce label overlap.
    dates = pd.to_datetime(data["date"])
    unique_dates = pd.Index(dates.drop_duplicates().sort_values())
    purge_horizon = horizon if mcfg.get("purge_horizon", True) else 0
    fold_count = max(2, int(mcfg.get("validation_folds", 3)))
    fold_fraction = float(mcfg.get("validation_fraction", 0.10))
    if len(unique_dates) < max(60, purge_horizon * fold_count + 30):
        return {}, {"enabled": True, "samples": int(len(data)), "auc": None, "error": "تاريخ غير كافٍ لاختبار زمني متعدد النوافذ"}

    rf = RandomForestClassifier(
        n_estimators=int(mcfg.get("rf_trees", 120)),
        max_depth=int(mcfg.get("rf_depth", 6)),
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    def make_deep():
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=128,
                learning_rate_init=0.001,
                max_iter=int(mcfg.get("mlp_max_iter", 120)),
                # Internal random early stopping can mix overlapping forward labels
                # across its split; external chronological folds decide model quality.
                early_stopping=False,
                random_state=42,
            )),
        ])

    fold_predictions = []
    fold_meta = []
    n_dates = len(unique_dates)
    fold_size = max(10, int(n_dates * fold_fraction))
    # Latest fold ends at the final available label; prior folds use the same width.
    first_start = n_dates - fold_size * fold_count
    if first_start <= purge_horizon + 20:
        first_start = max(int(n_dates * 0.50), purge_horizon + 21)
        fold_size = max(10, (n_dates - first_start) // fold_count)
    for fold in range(fold_count):
        start_idx = first_start + fold * fold_size
        end_idx = min(n_dates, start_idx + fold_size)
        train_end_idx = start_idx - purge_horizon
        if train_end_idx < 20 or end_idx - start_idx < 10:
            continue
        train_cutoff = unique_dates[train_end_idx]
        validation_start = unique_dates[start_idx]
        validation_end = unique_dates[end_idx - 1]
        train_mask = dates < train_cutoff
        validation_mask = (dates >= validation_start) & (dates <= validation_end)
        Xtr, ytr = X.loc[train_mask], y.loc[train_mask]
        Xva, yva = X.loc[validation_mask], y.loc[validation_mask]
        if len(Xtr) < int(mcfg.get("min_samples", 500)) * 0.35 or len(Xva) < 20:
            continue
        if ytr.nunique() < 2 or yva.nunique() < 2:
            continue
        fold_rf = RandomForestClassifier(
            n_estimators=int(mcfg.get("rf_trees", 120)),
            max_depth=int(mcfg.get("rf_depth", 6)),
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42 + fold,
            n_jobs=-1,
        )
        fold_deep = make_deep()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_rf.fit(Xtr, ytr)
            fold_deep.fit(Xtr, ytr)
        prob = 0.55 * fold_rf.predict_proba(Xva)[:, 1] + 0.45 * fold_deep.predict_proba(Xva)[:, 1]
        fold_predictions.append(pd.DataFrame({
            "y": yva.to_numpy(),
            "p": prob,
            "baseline": float(ytr.mean()),
            "fold": fold,
        }, index=yva.index))
        fold_meta.append({
            "fold": fold + 1,
            "train_samples": int(len(Xtr)),
            "validation_samples": int(len(Xva)),
            "validation_start": str(validation_start.date()),
            "validation_end": str(validation_end.date()),
        })

    if len(fold_predictions) < 2:
        return {}, {"enabled": True, "samples": int(len(data)), "auc": None, "error": "نوافذ الاختبار الزمني الصالحة أقل من نافذتين"}
    oof = pd.concat(fold_predictions).sort_index()
    yva = oof["y"].astype(int)
    p = oof["p"].astype(float)
    auc = float(roc_auc_score(yva, p))
    brier = float(brier_score_loss(yva, p))
    baseline_brier = float(brier_score_loss(yva, oof["baseline"].astype(float)))
    validation_meta = {
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "validation_samples": int(len(oof)),
        "validation_start": str(min(m["validation_start"] for m in fold_meta)),
        "validation_end": str(max(m["validation_end"] for m in fold_meta)),
        "validation_folds": fold_meta,
    }

    # معايرة احتمالات Platt: تُدرَّب على توقعات OOF، ولا تُطبَّق على الإنتاج
    # إلا إذا كانت Brier المعايرة أفضل من (أو مساوية لـ) الخام.
    calibrator = None
    calibration_meta = {"method": "platt", "applied": False}
    if bool(mcfg.get("calibration_enabled", True)):
        calibrator = _fit_platt(p.to_numpy(), yva.to_numpy())
        if calibrator is not None:
            p_cal = calibrator.predict_proba(p.to_numpy().reshape(-1, 1))[:, 1]
            brier_cal = float(brier_score_loss(yva, p_cal))
            calibration_meta.update({
                "oof_brier_raw": round(brier, 4),
                "oof_brier_calibrated": round(brier_cal, 4),
            })
            if brier_cal <= brier:
                calibration_meta["applied"] = True
            else:
                calibrator = None

    # لا نسمح للنموذج بتغيير ترتيب الفرص إلا إذا اجتاز AUC وحسّن
    # خط الأساس في Brier؛ جودة الترتيب وحدها لا تثبت جودة الاحتمالات.
    min_auc = float(mcfg.get("min_auc", 0.52))
    min_brier_improvement = float(mcfg.get("min_brier_improvement", 0.0))
    required_brier = baseline_brier * (1.0 - min_brier_improvement)
    if auc < min_auc or brier >= required_brier:
        reason = (
            "AUC أقل من الحد الأدنى؛ تم تجاهل النموذج"
            if auc < min_auc
            else "دقة الاحتمالات لم تتفوق على خط الأساس؛ تم تجاهل النموذج"
        )
        return {}, {
            "enabled": True, "samples": int(len(data)), **validation_meta,
            "model_used": False, "error": reason,
        }

    # Only after the walk-forward gate passes do we fit production models on all
    # available labelled history.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y)
    deep = make_deep()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deep.fit(X, y)

    latest_rows = []
    for code, f in frames.items():
        if len(f) == 0:
            continue
        row = f.iloc[-1]
        if not all(c in row.index for c in cols):
            continue
        vals = pd.DataFrame([[row[c] for c in cols]], columns=cols).replace([np.inf, -np.inf], np.nan)
        if vals.isna().any().any():
            continue
        pr = float(0.55 * rf.predict_proba(vals)[:, 1][0] + 0.45 * deep.predict_proba(vals)[:, 1][0])
        if calibrator is not None:
            pr = float(calibrator.predict_proba(np.array([[pr]]))[0, 1])
        latest_rows.append((code, pr))

    return dict(latest_rows), {
        "enabled": True, "samples": int(len(data)), **validation_meta,
        "model_used": True, "model": "RandomForest + MLP(32,16)",
        "calibration": calibration_meta,
        "horizon_days": horizon,
        "net_positive_return": net_threshold,
        "round_trip_cost_pct": cost_pct,
        "gross_return_threshold": gross_threshold,
    }

