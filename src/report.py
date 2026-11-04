"""التقرير اليومي.

المبدأ: الكود يحسب كل الأرقام والمستويات، والذكاء الاصطناعي يكتب التعليق فقط.
وأي تعليق فيه رقم مش موجود في بيانات اليوم بيتم تجاهله ويحل محله نص جاهز من الكود.
"""
from __future__ import annotations

import html
import json
import os
import re
from datetime import date, datetime

import requests

AR_DAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]

REGIME_ADVICE = {
    "strong_up": "البيئة داعمة لفرص الشراء الانتقائية.",
    "up": "البيئة داعمة لفرص الشراء الانتقائية.",
    "neutral": "البيئة محايدة، فالانتقائية أهم وحجم المراكز يفضّل يكون أصغر.",
    "weak": "البيئة ضعيفة، والحذر أولى مع أقل عدد ممكن من الصفقات.",
    "down": "السوق هابط، فلا توصيات شراء اليوم والأفضل الانتظار.",
}

DISCLAIMER = (
    "⚠️ هذا تحليل آلي للاستخدام الشخصي وليس نصيحة استثمارية، والأداء السابق لا يضمن المستقبل. "
    "الأسعار من آخر جلسة متاحة؛ لو افتتح السهم فوق منطقة الدخول بفجوة كبيرة فتجنب المطاردة. "
    "النطاق يشمل مكونات EGX33 وأسهمًا خارج المؤشر لها مصدر فرز شرعي مسجل وحديث؛ انتهاء المراجعة يمنع إشارة الشراء حتى تجديدها. "
    "الإعداد يطلب تاريخًا يصل إلى 8 سنوات، لكن التغطية الفعلية تختلف حسب الرمز والمصدر وتظهر في صفحة عن النظام. "
    "المؤشر الشرعي الداخلي بديل متساوي الأوزان من المكونات الحالية وليس سجلًا رسميًا تاريخيًا لمكونات EGX33."
)

SYSTEM_PROMPT = (
    "أنت محلل فني للبورصة المصرية تكتب تعليقًا مختصرًا لمستثمر فرد. ستستلم بيانات محسوبة بالكود بصيغة JSON.\n"
    "القواعد:\n"
    "1) استخدم الأرقام الموجودة في البيانات فقط. لا تخترع أي رقم أو خبر أو معلومة عن الشركات أو الاقتصاد أو القطاعات.\n"
    "2) لا تعطِ وعودًا أو ضمانات ولا تستخدم عبارات يقين مثل (مضمون) أو (أكيد).\n"
    "3) لكل سهم: جملتان كحد أقصى، الأولى عن سبب الترشيح والثانية عن الخطر الرئيسي.\n"
    "4) لو حالة السوق ضعيفة أو هابطة فاذكر أن الحذر أولى.\n"
    "5) الأخبار مجرد عناوين وسياق؛ لا تستنتج منها أرقامًا أو أحداثًا غير مكتوبة، ولا تحول الخبر وحده إلى توصية.\n"
    "6) اكتب بعربية مبسطة وواضحة بدون Markdown أو HTML أو رموز تنسيق.\n"
    "7) أرجع JSON فقط بهذا الشكل بالضبط: "
    '{"market_view": "ملخص السوق في 2-3 جمل", "picks": {"رمز السهم": "تعليق"}, "watch_view": "جملة واحدة عن قائمة المراقبة"}'
)


# ------------------------------------------------------------------ أدوات صغيرة
def ar_date(d: date | datetime) -> str:
    return f"{AR_DAYS[d.weekday()]} {d.day} {AR_MONTHS[d.month - 1]} {d.year}"


def esc(x) -> str:
    return html.escape(str(x), quote=False)


def fmt(x, nd: int = 2) -> str:
    return "-" if x is None else f"{float(x):.{nd}f}"


def arrow(x) -> str:
    if x is None:
        return "-"
    x = float(x)
    return f"▲ {abs(x):.2f}%" if x > 0 else f"▼ {abs(x):.2f}%" if x < 0 else "◀▶ 0.00%"


# ------------------------------------------------------------------ بيانات الذكاء الاصطناعي
def build_payload(result: dict, track: dict | None) -> dict:
    """نسخة مختصرة من نتيجة اليوم. ده كل اللي الذكاء الاصطناعي يشوفه."""
    m = result["market"]

    def slim(p: dict) -> dict:
        keys = ["code", "name", "score", "setup_ar", "close", "entry_low", "entry_high", "stop", "t1", "t2",
                "risk_pct", "rsi", "rs_pct", "vol_ratio", "ret20", "why", "cautions", "why_not_buy"]
        return {k: p.get(k) for k in keys}

    return {
        "data_date": result["data_date"],
        "regime": result["regime"],
        "market": {k: m.get(k) for k in [
            "bench_name", "bench_change_1d", "bench_ret20", "breadth50", "breadth200",
            "advancers", "decliners", "top_gainers", "top_losers"]},
        "picks": [slim(p) for p in result["picks"]],
        "watch": [slim(p) for p in result["watch"][:5]],
        "track_record": (track or {}).get("stats"),
        "funds": result.get("funds", {}),
        "ml": result.get("ml", {}),
        "news": {
            "market": [{"title": x.get("title"), "source": x.get("source")} for x in (result.get("news", {}).get("market") or [])[:6]],
            "companies": [{"code": x.get("code"), "title": x.get("title"), "source": x.get("source")} for x in (result.get("news", {}).get("companies") or [])[:10]],
        },
        "portfolio": {
            "summary": result.get("portfolio", {}).get("summary", {}),
            "decisions": [
                {"النوع": x.get("النوع"), "الأصل": x.get("الأصل"), "القرار": x.get("القرار"),
                 "الربح/الخسارة %": x.get("الربح/الخسارة %"), "النقاط": x.get("النقاط"), "reason": x.get("reason")}
                for x in (result.get("portfolio", {}).get("decisions") or [])
            ],
        },
    }


_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
_ALWAYS_OK = {"0", "1", "2", "3", "4", "5", "10", "14", "20", "50", "52", "60", "100", "200"}


def allowed_numbers(payload) -> set[str]:
    out = set(_ALWAYS_OK)

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            for nd in (0, 1, 2):
                out.add(f"{abs(float(v)):.{nd}f}")
            out.add(str(abs(int(v))))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            for tok in re.findall(r"\d+(?:\.\d+)?", v.translate(_AR_DIGITS)):
                out.add(tok)

    walk(payload)
    return out


def numbers_ok(text: str, allowed: set[str]) -> bool:
    for tok in re.findall(r"\d+(?:[.,]\d+)?", text.translate(_AR_DIGITS)):
        tok = tok.replace(",", ".")
        norm = tok.rstrip("0").rstrip(".") if "." in tok else tok
        if tok not in allowed and norm not in allowed:
            return False
    return True


# ------------------------------------------------------------------ استدعاء النموذج
def _extract_json(text: str) -> dict | None:
    """استخراج JSON من Gemini حتى لو أضاف Markdown أو نصًا حوله."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    try:
        import ast
        obj = ast.literal_eval(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def call_llm(payload: dict, cfg: dict, post=requests.post, env=os.environ, system: str | None = None) -> tuple[dict | None, dict]:
    """يرجع (تعليقات أو None، معلومات عن التشغيلة). أي فشل = None والتقرير يكمل بالنص الجاهز.

    `system` يسمح بمهمة مختلفة (مثل تصنيف مشاعر العناوين) بنفس المزود والمفاتيح؛
    الافتراضي هو مطبع التعليقات اليومي.
    """
    l = cfg["llm"]
    provider = (l.get("provider") or "none").lower()
    meta = {"provider": provider, "ok": False, "error": ""}
    if provider == "none":
        meta["error"] = "معطّل من الإعدادات"
        return None, meta
    user = json.dumps(payload, ensure_ascii=False)
    sys_prompt = system or SYSTEM_PROMPT
    try:
        if provider == "anthropic":
            key = env.get("ANTHROPIC_API_KEY", "")
            if not key:
                meta["error"] = "ANTHROPIC_API_KEY غير مضبوط"
                return None, meta
            meta["model"] = l["anthropic_model"]
            r = post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": l["anthropic_model"], "max_tokens": l["max_tokens"], "system": sys_prompt,
                      "messages": [{"role": "user", "content": user}]},
                timeout=l["timeout"],
            )
            if r.status_code != 200:
                meta["error"] = f"HTTP {r.status_code}"
                return None, meta
            text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        elif provider == "gemini":
            key = env.get("GEMINI_API_KEY", "")
            if not key:
                meta["error"] = "GEMINI_API_KEY غير مضبوط"
                return None, meta
            meta["model"] = l["gemini_model"]
            r = post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{l['gemini_model']}:generateContent",
                params={"key": key},
                json={"systemInstruction": {"parts": [{"text": sys_prompt}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"maxOutputTokens": max(int(l["max_tokens"]), 3000), "responseMimeType": "application/json"}},
                timeout=l["timeout"],
            )
            if r.status_code != 200:
                meta["error"] = f"HTTP {r.status_code}"
                return None, meta
            body = r.json()
            candidates = body.get("candidates") or []
            if not candidates:
                meta["error"] = "Gemini لم يُرجع candidates"
                return None, meta
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(str(part.get("text", "")) for part in parts if part.get("text"))
            if not text:
                meta["error"] = "Gemini بلا نص (finishReason=%s)" % candidates[0].get("finishReason", "unknown")
                return None, meta
        else:
            meta["error"] = f"provider غير معروف: {provider}"
            return None, meta
    except Exception as e:  # noqa: BLE001
        meta["error"] = f"{type(e).__name__}"
        return None, meta

    data = _extract_json(text)
    if not isinstance(data, dict):
        meta["error"] = "رد غير قابل للقراءة"
        return None, meta
    meta["ok"] = True
    return data, meta


# ------------------------------------------------------------------ النصوص الجاهزة (بدون ذكاء اصطناعي)
def fallback_notes(result: dict) -> dict:
    """نصوص جاهزة من الكود. أرقام السوق موجودة في سطر الحالة نفسه، فهنا نضيف الخلاصة بس."""
    rg = result["regime"]
    view = REGIME_ADVICE.get(rg["label"], "")
    if rg["label"] != "down":
        view += f" الحد الأقصى لفرص الشراء اليوم {rg['max_buys']}."
    return {
        "market_view": view.strip(),
        "picks": {p["code"]: "؛ ".join(p.get("why", [])[:2]) for p in result["picks"]},
        "watch_view": "أسهم تستحق المتابعة لكنها لم تستوفِ كل شروط الشراء بعد.",
    }


def merge_notes(llm: dict | None, fb: dict, payload: dict) -> tuple[dict, int]:
    """يدمج تعليقات النموذج مع النص الجاهز ويسقط أي تعليق فيه أرقام غريبة. يرجع (تعليقات، عدد المسقط)."""
    if not llm:
        return fb, 0
    allowed = allowed_numbers(payload)
    dropped = 0
    out = {"market_view": fb["market_view"], "picks": dict(fb["picks"]), "watch_view": fb["watch_view"]}

    def take(text) -> str | None:
        nonlocal dropped
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()[:500]
        if not numbers_ok(text, allowed):
            dropped += 1
            return None
        return text

    mv = take(llm.get("market_view"))
    if mv:
        out["market_view"] = mv
    wv = take(llm.get("watch_view"))
    if wv:
        out["watch_view"] = wv
    if isinstance(llm.get("picks"), dict):
        for code in out["picks"]:
            t = take(llm["picks"].get(code))
            if t:
                out["picks"][code] = t
    return out, dropped


# ------------------------------------------------------------------ رسالة تلجرام
def compose_message(result: dict, notes: dict, track: dict | None, cfg: dict, warnings: list[str], now: datetime) -> str:
    rg, m, tg = result["regime"], result["market"], cfg["telegram"]
    data_d = date.fromisoformat(result["data_date"])
    paras: list[str] = []

    paras.append(f"📊 <b>تقرير البورصة المصرية</b>\n{ar_date(now)}\nبيانات إغلاق {ar_date(data_d)}")

    lines = [f"<b>حالة السوق: {esc(rg['label_ar'])}</b> ({rg['score']:.0f}/100)"]
    stat = [f"{esc(m['bench_name'])} {arrow(m.get('bench_change_1d'))}"]
    if m.get("breadth50") is not None:
        stat.append(f"فوق م.50: {m['breadth50']:.0f}%")
    stat.append(f"صاعد {m['advancers']} / هابط {m['decliners']}")
    lines.append(" | ".join(stat))
    if m.get("top_gainers") and m.get("top_losers"):
        g, lo = m["top_gainers"][0], m["top_losers"][0]
        lines.append(f"الأقوى: {esc(g['code'])} {arrow(g['ret1'])} | الأضعف: {esc(lo['code'])} {arrow(lo['ret1'])}")
    lines.append(esc(notes["market_view"]))
    paras.append("\n".join(lines))

    picks = result["picks"][: tg["max_picks"]]
    if picks:
        paras.append(f"✅ <b>فرص الشراء ({len(picks)})</b>")
        for i, p in enumerate(picks, 1):
            block = [
                f"<b>{i}. {esc(p['code'])}</b> — {esc(p['name'])} · نقاط {p['score']:.0f}",
                f"النموذج: {esc(p['setup_ar'])}",
                f"الدخول: من {fmt(p['entry_low'])} إلى {fmt(p['entry_high'])}",
                f"الوقف: {fmt(p['stop'])} (مخاطرة {fmt(p['risk_pct'], 1)}%)",
                f"الهدف 1: {fmt(p['t1'])} | الهدف 2: {fmt(p['t2'])}",
            ]
            if p.get("position_pct") is not None:
                block.append(f"حجم المركز المقترح: حتى {p['position_pct']:.0f}% من المحفظة")
            note = notes["picks"].get(p["code"])
            if note:
                block.append(esc(note))
            if p.get("cautions"):
                block.append(f"⚠️ {esc(p['cautions'][0])}")
            paras.append("\n".join(block))
    else:
        why = "السوق هابط ولا توصيات شراء." if rg["max_buys"] == 0 else "لا سهم استوفى كل شروط الشراء اليوم."
        paras.append(f"✅ <b>فرص الشراء</b>\nلا توصيات شراء اليوم. {esc(why)}")

    watch = result["watch"][: tg["max_watch"]]
    if watch:
        rows = []
        for p in watch:
            reason = (p.get("why_not_buy") or [""])[0]
            rows.append(f"• <b>{esc(p['code'])}</b> ({p['score']:.0f})" + (f" — {esc(reason)}" if reason else ""))
        paras.append("👀 <b>قائمة المراقبة</b>\n" + esc(notes["watch_view"]) + "\n" + "\n".join(rows))

    portfolio = result.get("portfolio") or {}
    decisions = portfolio.get("decisions") or []
    if decisions:
        rows = []
        for x in decisions:
            pnl = x.get("الربح/الخسارة %")
            pnl_txt = f" | P/L {float(pnl):+.1f}%" if pnl is not None else ""
            rows.append(f"• <b>{esc(x.get('الأصل',''))}</b>: <b>{esc(x.get('القرار',''))}</b>{pnl_txt} — {esc(x.get('reason',''))}")
        paras.append("💼 <b>قرارات محفظتك</b>\n" + "\n".join(rows))

    news = result.get("news") or {}
    market_news = news.get("market") or []
    if market_news:
        rows = []
        for x in market_news[:4]:
            title = esc(x.get("title", ""))
            source = esc(x.get("source", ""))
            rows.append(f"• {title}" + (f" — {source}" if source else ""))
        paras.append("📰 <b>أهم أخبار السوق</b>\n" + "\n".join(rows))

    company_news = news.get("companies") or []
    if company_news:
        rows = []
        for x in company_news[:6]:
            code = esc(x.get("code", ""))
            rows.append(f"• <b>{code}</b> {esc(x.get('title', ''))}")
        paras.append("🏢 <b>أخبار مرتبطة بالفرص</b>\n" + "\n".join(rows))

    if track and track.get("closed"):
        s = track["stats"]
        small = " (عينة صغيرة)" if track["closed"] < 20 else ""
        paras.append(
            f"📈 <b>سجل توصياتنا</b>{small}\n"
            f"مغلقة {track['closed']} | نجاح {s.get('win_rate', 0):.0f}% | متوسط {s.get('avg_r', 0):+.2f}R | مفتوحة {track['open']}"
        )

    if warnings:
        paras.append("🔧 <b>ملاحظات على البيانات</b>\n" + "\n".join(f"• {esc(w)}" for w in warnings))

    paras.append(esc(DISCLAIMER))
    return "\n\n".join(paras)


def compose_alert(reason: str) -> str:
    return f"⚠️ <b>تعذّر إنشاء تقرير اليوم</b>\n{esc(reason[:600])}\nراجع تبويب Actions في GitHub لمعرفة التفاصيل."


# ------------------------------------------------------------------ نقطة الدخول
def build_report(result: dict, track: dict | None, cfg: dict, now: datetime, warnings: list[str],
                 post=requests.post, env=os.environ) -> dict:
    payload = build_payload(result, track)
    llm, meta = call_llm(payload, cfg, post=post, env=env)
    notes, dropped = merge_notes(llm, fallback_notes(result), payload)
    meta["dropped_notes"] = dropped
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "data_date": result["data_date"],
        "notes": notes,
        "llm": meta,
        "warnings": warnings,
        "telegram_html": compose_message(result, notes, track, cfg, warnings, now),
    }
