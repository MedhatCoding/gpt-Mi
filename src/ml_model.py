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


# ------------------------------------------------------------------ النموذج العميق (اختياري)
def _deep_model_status(span_days: int, mcfg: dict) -> dict:
    """بوابة النموذج العميق (LSTM + attention): لا يشتغل إلا بسنتين تاريخ على
    الأقل وTensorFlow مثبت؛ غير مثبت في CI اليومي عمدًا ليبقى سريعًا وخفيفًا."""
    d = mcfg.get("deep_model") or {}
    out = {"eligible": False, "available": False, "used": False, "reason": ""}
    if not d.get("enabled", False):
        out["reason"] = "النموذج العميق معطل من الإعدادات"
        return out
    min_days = int(d.get("min_history_days", 730))
    if span_days < min_days:
        out["reason"] = (
            f"التاريخ المتاح {span_days} يوم أقل من {min_days} يوم (سنتان تقريبًا)؛ "
            "النموذج العميق مؤجل حتى يطول التاريخ"
        )
        return out
    out["eligible"] = True
    try:
        import tensorflow  # noqa: F401
    except Exception:
        out["reason"] = "TensorFlow غير مثبت (requirements-deep.txt اختياري)؛ الطبقة العميقة متأجلة"
        return out
    out["available"] = True
    return out


def _fit_lstm(train_data: pd.DataFrame, cols: list[str], dcfg: dict):
    """يدرب LSTM صغير + طبقة attention على تتابعات الميزات لكل سهم.

    يرجع (model, scaler, seq_days) أو None عند أي نقص (مكتبة/عينات/فئات).
    التتابعات تُبنى من صفوف data المتاحة؛ أي فجوات صفوف مسقوطة (NaN) تُقرَّب
    بصفر بعد التقييس — تقريب مقصود وواضح، والبوابة تقيس أثره فعليًا على OOF.
    """
    try:
        from tensorflow.keras import layers, models  # type: ignore
    except Exception:
        return None
    seq_days = int(dcfg.get("sequence_days", 20))
    units = int(dcfg.get("units", 24))
    max_rows = int(dcfg.get("max_train_rows", 40000))
    min_rows = int(dcfg.get("min_train_rows", 1000))
    scaler = StandardScaler().fit(train_data[cols].astype(float))
    Xs, ys = [], []
    for _code, sub in train_data.groupby("code", sort=False):
        arr = scaler.transform(sub[cols].astype(float)).astype("float32")
        tgt = sub["target"].to_numpy()
        for i in range(seq_days - 1, len(arr)):
            Xs.append(arr[i - seq_days + 1:i + 1])
            ys.append(tgt[i])
    if len(Xs) < max(200, min_rows) or len(np.unique(ys)) < 2:
        return None
    X = np.stack(Xs)
    Y = np.asarray(ys, dtype="float32")
    if len(X) > max_rows:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(X), size=max_rows, replace=False)
        X, Y = X[keep], Y[keep]
    inp = layers.Input(shape=(seq_days, len(cols)))
    seq = layers.LSTM(units, return_sequences=True)(inp)
    attn = layers.Attention()([seq, seq])
    pooled = layers.GlobalAveragePooling1D()(attn)
    out = layers.Dense(1, activation="sigmoid")(pooled)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X, Y, epochs=int(dcfg.get("epochs", 8)), batch_size=256, verbose=0)
    return model, scaler, seq_days


def _lstm_fold_probs(data: pd.DataFrame, cols: list[str], train_mask: pd.Series,
                     pred_mask: pd.Series, dcfg: dict) -> dict[int, float]:
    """احتمالات LSTM لصفوف pred_mask بعد تدريب على train_mask فقط (داخل نوافذ walk-forward)."""
    fitted = _fit_lstm(data.loc[train_mask], cols, dcfg)
    if fitted is None:
        return {}
    model, scaler, seq_days = fitted
    seqs, rows = [], []
    for code, sub_full in data.groupby("code", sort=False):
        arr = scaler.transform(sub_full[cols].astype(float)).astype("float32")
        pos_map = {row: i for i, row in enumerate(sub_full.index)}
        for row in sub_full.loc[pred_mask.loc[sub_full.index]].index:
            i = pos_map.get(row)
            if i is None or i < seq_days - 1:
                continue
            seqs.append(arr[i - seq_days + 1:i + 1])
            rows.append(int(row))
    if not seqs:
        return {}
    probs = model.predict(np.stack(seqs), verbose=0).ravel()
    return {r: float(p) for r, p in zip(rows, probs)}


def _lstm_latest_probs(data: pd.DataFrame, cols: list[str], frames: dict[str, pd.DataFrame],
                       dcfg: dict) -> dict[str, float]:
    """احتمالات LSTM لآخر seq_days شمعة من كل إطار بعد التدريب على كل البيانات المسماة."""
    fitted = _fit_lstm(data, cols, dcfg)
    if fitted is None:
        return {}
    model, scaler, seq_days = fitted
    out: dict[str, float] = {}
    for code, f in frames.items():
        if len(f) == 0:
            continue
        vals = f[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = scaler.transform(vals.astype(float)).astype("float32")
        if len(arr) < seq_days:
            continue
        seq = arr[-seq_days:][None, ...]
        out[str(code)] = float(model.predict(seq, verbose=0).ravel()[0])
    return out


def permutation_importance(Xoof: pd.DataFrame, y: pd.Series, prob_fn, n_repeats: int = 2,
                           seed: int = 42) -> list[dict]:
    """أثر كل مؤشر على AUC عبر خلط قيمه (permutation importance) على عينات التحقق.

    تفسير للعرض فقط: أي مؤشر درجته موجبة يعني أن إخفاءه يخفض قدرة الترتيب.
    """
    base_auc = float(roc_auc_score(y, prob_fn(Xoof)))
    rng = np.random.default_rng(seed)
    out = []
    for col in Xoof.columns:
        drops = []
        for _ in range(max(1, int(n_repeats))):
            Xp = Xoof.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base_auc - float(roc_auc_score(y, prob_fn(Xp))))
        out.append({"feature": col, "auc_drop": round(float(np.mean(drops)), 4)})
    return sorted(out, key=lambda r: -r["auc_drop"])


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

    # بوابة النموذج العميق: لا يُفعَّل إلا بسنتين تاريخ على الأقل وTensorFlow
    # مثبت (requirements-deep.txt اختياري). في CI اليومي يتأجل بأمان بلا أي
    # تغيير في السلوك، وعلى جهاز مثبت عليه TensorFlow تُقاس إضافته على OOF أولًا.
    span_days = int((unique_dates[-1] - unique_dates[0]).days) if len(unique_dates) else 0
    dcfg = dict(mcfg.get("deep_model") or {})
    lstm_status = _deep_model_status(span_days, mcfg)
    use_lstm = bool(lstm_status["eligible"] and lstm_status["available"])

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
        fold_row = {
            "y": yva.to_numpy(),
            "p": prob,
            "baseline": float(ytr.mean()),
            "fold": fold,
        }
        # توقعات الطبقة العميقة للنافذة (اختياري): أي فشل يعطلها لباقي التدريبات
        # ولا يؤثر على مسار RF+MLP الأساسي.
        if use_lstm:
            try:
                lstm_probs = _lstm_fold_probs(data, cols, train_mask, validation_mask, dcfg)
            except Exception as e:  # noqa: BLE001
                use_lstm = False
                lstm_status["reason"] = f"فشل تدريب الطبقة العميقة: {type(e).__name__}"
                lstm_probs = {}
            if lstm_probs:
                fold_row["p_deep"] = [lstm_probs.get(int(i)) for i in yva.index]
        fold_predictions.append(pd.DataFrame(fold_row, index=yva.index))
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
    # دمج اختياري للطبقة العميقة: يُقبل فقط لو حسّن AUC على العينة الخارجية،
    # وبعدها تعبر البوابات والمعايرة على الاحتمال المدموج الفعلي.
    lstm_blend_weight = 0.0
    if use_lstm and "p_deep" in oof.columns:
        dmask = oof["p_deep"].notna()
        min_oof_rows = int(dcfg.get("min_oof_rows", 200))
        if int(dmask.sum()) >= min_oof_rows and oof.loc[dmask, "y"].nunique() == 2:
            yd = oof.loc[dmask, "y"].astype(int)
            base_p = oof.loc[dmask, "p"].astype(float)
            deep_p = oof.loc[dmask, "p_deep"].astype(float)
            w = float(dcfg.get("blend_weight", 0.20))
            auc_base = float(roc_auc_score(yd, base_p))
            auc_blend = float(roc_auc_score(yd, (1 - w) * base_p + w * deep_p))
            lstm_status["oof_auc_lstm"] = round(float(roc_auc_score(yd, deep_p)), 4)
            if auc_blend > auc_base:
                lstm_blend_weight = w
                lstm_status["used"] = True
                lstm_status["oof_auc_base"] = round(auc_base, 4)
                lstm_status["oof_auc_blended"] = round(auc_blend, 4)
                oof.loc[dmask, "p"] = ((1 - w) * base_p + w * deep_p).to_numpy()
            else:
                lstm_status["reason"] = "الطبقة العميقة لم تحسّن AUC خارج العينة؛ استُبعدت"
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

    # تفسير قرارات النموذج (permutation importance) على عينات التحقق: أي مؤشر
    # إخفاؤه يخفض AUC هو دافع حقيقي للترتيب. تقدير تفسيري للعرض فقط.
    importance_meta = {"method": "permutation", "enabled": False, "note": "تقدير تفسيري على عينات التحقق بنماذج الإنتاج؛ للعرض لا للقرار."}
    imp_rows: list[dict] = []
    icfg = mcfg.get("importance") or {}
    # الافتراضي معطل: كتلة التفسير مكلفة نسبيًا وتُفعَّل في الإنتاج فقط عبر
    # config.yaml (ml.importance.enabled) حتى لا تُشغل في كل استدعاء اختباري.
    if icfg.get("enabled", False) and len(oof) >= 200:
        try:
            Xoof = X.loc[oof.index]
            yoof = oof["y"].astype(int)

            def prob_fn(D):
                return 0.55 * rf.predict_proba(D)[:, 1] + 0.45 * deep.predict_proba(D)[:, 1]

            ranked = permutation_importance(Xoof, yoof, prob_fn, n_repeats=int(icfg.get("n_repeats", 2)))
            top_n = int(icfg.get("top_n", 8))
            imp_rows = [r for r in ranked if float(r["auc_drop"]) > 0][:top_n]
            importance_meta["enabled"] = True
        except Exception:  # noqa: BLE001
            imp_rows = []

    # توقعات الطبقة العميقة للإنتاج (لو قُبل دمجها على OOF).
    lstm_latest: dict[str, float] = {}
    if lstm_blend_weight:
        try:
            lstm_latest = _lstm_latest_probs(data, cols, frames, dcfg)
        except Exception:  # noqa: BLE001
            lstm_latest = {}

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
        if lstm_blend_weight and str(code) in lstm_latest:
            pr = float((1 - lstm_blend_weight) * pr + lstm_blend_weight * lstm_latest[str(code)])
        if calibrator is not None:
            pr = float(calibrator.predict_proba(np.array([[pr]]))[0, 1])
        latest_rows.append((code, pr))

    lstm_status["blend_weight"] = lstm_blend_weight
    return dict(latest_rows), {
        "enabled": True, "samples": int(len(data)), **validation_meta,
        "model_used": True,
        "model": "RandomForest + MLP(32,16)" + (" + LSTM(attention)" if lstm_blend_weight else ""),
        "calibration": calibration_meta,
        "horizon_days": horizon,
        "net_positive_return": net_threshold,
        "round_trip_cost_pct": cost_pct,
        "gross_return_threshold": gross_threshold,
        "deep_model": lstm_status,
        "feature_importance": imp_rows,
        "interpretation": importance_meta,
    }

