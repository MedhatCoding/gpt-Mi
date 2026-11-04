"""مصدر أخبار خفيف للتقرير اليومي: Google News RSS بدون مفتاح، مع حماية من فشل المصدر.
لا نستخدم الأخبار كرقم داخل نموذج التداول؛ هي سياق للمستثمر فقط.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

BASE = "https://news.google.com/rss/search"
UA = "Mozilla/5.0 EGX-Advisor/1.0"

def _fetch(query: str, limit: int = 8, timeout: int = 15) -> list[dict]:
    params = {"q": f"{query} when:2d", "hl": "ar", "gl": "EG", "ceid": "EG:ar"}
    try:
        r = requests.get(BASE, params=params, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for item in root.findall(".//item")[:limit]:
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        source_el = item.find("source")
        source = html.unescape((source_el.text or "").strip()) if source_el is not None else ""
        pub = item.findtext("pubDate") or ""
        dt = None
        try:
            dt = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        title = re.sub(r"\s+", " ", title)
        if title:
            out.append({"title": title, "source": source, "published": dt, "url": link})
    return out

def fetch_news(picks: list[dict] | None = None, market_limit: int = 6, pick_limit: int = 2) -> dict:
    """يجلب أخبار السوق، ثم أخبار أهم فرص الشراء فقط لتجنب كثرة الطلبات."""
    market = _fetch("البورصة المصرية EGX OR EGX30 OR الأسهم المصرية", market_limit)
    companies = []
    seen = set()
    for p in (picks or [])[:5]:
        name = str(p.get("name") or "").strip()
        code = str(p.get("code") or "").strip()
        q = f'"{name}" OR {code} البورصة'
        for item in _fetch(q, pick_limit):
            key = item["title"]
            if key not in seen:
                seen.add(key)
                item["code"] = code
                companies.append(item)
    return {"market": market, "companies": companies, "generated_at": datetime.now(timezone.utc).isoformat()}

def compact_for_llm(news: dict) -> dict:
    return {
        "market": [{"title": x["title"], "source": x.get("source")} for x in news.get("market", [])[:6]],
        "companies": [{"code": x.get("code"), "title": x["title"], "source": x.get("source")} for x in news.get("companies", [])[:10]],
    }



# ------------------------------------------------------------------ مشاعر العناوين
# تصنيف معجمي مبسط (lexicon) للعناوين العربية والإنجليزية؛ تقدير تقريبي
# للعرض فقط ولا يدخل في أي قرار رقمي أو نموذج تداول.
POSITIVE_TERMS = (
    "ارتفاع", "صعود", "نمو", "أرباح", "توزيعات", "توسع", "زيادة", "قياسي",
    "surge", "profit", "growth", "gain", "record", "rise", "boost",
)
NEGATIVE_TERMS = (
    "خسارة", "خسائر", "هبوط", "تراجع", "توقف", "تأجيل", "غرامة",
    "انخفاض", "خفض", "نزاع", "loss", "drop", "fall", "decline", "delay", "suspend",
)


def score_headline(title: str) -> int:
    """درجة معجمية للعنوان: موجب يزيد وسالب ينقص. ليست تحليل مشاعر دقيقًا."""
    t = str(title or "").lower()
    return sum(1 for w in POSITIVE_TERMS if w in t) - sum(1 for w in NEGATIVE_TERMS if w in t)


# ------------------------------------------------------------------ مشاعر عبر النموذج اللغوي
# ترقية اختيارية فوق القاموس: نفس مزود تعليقات التقرير ونفس المفاتيح، بمهمة تصنيف
# مخصصة. أي فشل يعود تلقائيًا للتصنيف المعجمي فلا يتوقف التقرير أبدًا.
SENTIMENT_SYSTEM_PROMPT = (
    "أنت محلل أخبار مالي مصري. مهمتك تصنيف مشاعر عناوين الأخبار فقط. ستستلم JSON فيه عناوين مرقمة: "
    "M1..Mn لأخبار السوق وC1..Cn لأخبار الشركات.\n"
    "القواعد:\n"
    "1) صنّف كل عنوان بدرجة من -2 إلى +2: موجب قوي 2، موجب 1، محايد 0، سالب -1، سالب قوي -2.\n"
    "2) صنّف نص العنوان المكتوب فقط؛ لا تخترع أحداثًا أو معلومات غير مذكورة فيه.\n"
    "3) العنوان المزدوج أو غير الواضح درجته 0.\n"
    "4) أرجع JSON فقط بهذا الشكل بالضبط: "
    '{"market": {"M1": 0}, "companies": {"C1": 0}}'
)


def _sentiment_payload(news: dict, max_headlines: int = 12) -> dict:
    market = [
        {"id": f"M{i + 1}", "title": x.get("title", "")}
        for i, x in enumerate((news.get("market") or [])[:max_headlines])
    ]
    companies = [
        {"id": f"C{i + 1}", "code": x.get("code"), "title": x.get("title", "")}
        for i, x in enumerate((news.get("companies") or [])[:max_headlines])
    ]
    return {"market_headlines": market, "company_headlines": companies}


def parse_llm_scores(data: dict, payload: dict) -> dict | None:
    """يحوّل رد النموذج لدرجات موثقة؛ يرجع None لو أقل من نصف العناوين قابلة للقراءة."""
    def _score(v) -> int | None:
        try:
            s = int(round(float(v)))
        except Exception:
            return None
        return s if -2 <= s <= 2 else None

    raw = data.get("scores") if isinstance(data.get("scores"), dict) else data
    if not isinstance(raw, dict):
        return None
    m_raw = raw.get("market") if isinstance(raw.get("market"), dict) else {}
    c_raw = raw.get("companies") if isinstance(raw.get("companies"), dict) else {}
    market_scores: dict[str, int] = {}
    company_scores: dict[str, int] = {}
    for item in payload.get("market_headlines") or []:
        s = _score(m_raw.get(item["id"]))
        if s is not None:
            market_scores[item["id"]] = s
    for item in payload.get("company_headlines") or []:
        s = _score(c_raw.get(item["id"]))
        if s is not None:
            company_scores[item["id"]] = s
    total = len(payload.get("market_headlines") or []) + len(payload.get("company_headlines") or [])
    if not total or len(market_scores) + len(company_scores) < max(1, total // 2):
        return None
    return {"market": market_scores, "companies": company_scores}


def llm_sentiment_summary(news: dict, cfg: dict) -> dict | None:
    """مشاعر العناوين عبر النموذج اللغوي المضبوط في llm.provider.

    أي فشل (مفتاح غايب، انقطاع، رد غير قابل للقراءة) يرجع None؛ المستدعي يرجع للقاموس.
    """
    try:
        scfg = ((cfg or {}).get("llm") or {}).get("news_sentiment") or {}
        if not scfg.get("enabled", True):
            return None
        from .report import call_llm  # استيراد محلي عمدًا: يمنع أي دورة استيراد مستقبلية
        payload = _sentiment_payload(news, max_headlines=int(scfg.get("max_headlines", 12)))
        if not payload["market_headlines"] and not payload["company_headlines"]:
            return None
        data, meta = call_llm(payload, cfg, system=SENTIMENT_SYSTEM_PROMPT)
        if not meta.get("ok") or not isinstance(data, dict):
            return None
        scores = parse_llm_scores(data, payload)
        if scores is None:
            return None
        market_vals = [scores["market"][item["id"]] for item in payload["market_headlines"] if item["id"] in scores["market"]]
        per_code: dict[str, list[int]] = {}
        for item in payload["company_headlines"]:
            code = str(item.get("code") or "").upper()
            s = scores["companies"].get(item["id"])
            if code and s is not None:
                per_code.setdefault(code, []).append(s)
        return {
            "method": "llm",
            "model": meta.get("model"),
            "market": round(sum(market_vals) / len(market_vals), 2) if market_vals else None,
            "companies": {c: round(sum(v) / len(v), 2) for c, v in per_code.items()},
            "note": "مشاعر مقدرة بنموذج لغوي للعناوين فقط؛ تقدير عرضي لا يدخل في أي قرار رقمي ولا في نموذج التداول.",
        }
    except Exception:
        return None


def sentiment_summary(news: dict, cfg: dict | None = None) -> dict:
    """متوسط درجات العناوين للسوق ولكل سهم من المختارات.

    لو مررنا الإعدادات: تجربة النموذج اللغوي أولًا، والرجوع للقاموس عند أي فشل.
    """
    if cfg is not None:
        llm = llm_sentiment_summary(news, cfg)
        if llm is not None:
            return llm

    def avg(items: list[dict]):
        scores = [score_headline(x.get("title", "")) for x in items]
        return round(sum(scores) / len(scores), 2) if scores else None

    per_code: dict[str, list[dict]] = {}
    for x in news.get("companies", []) or []:
        code = str(x.get("code") or "").upper()
        if code:
            per_code.setdefault(code, []).append(x)
    return {
        "method": "lexicon",
        "market": avg(news.get("market", []) or []),
        "companies": {c: avg(items) for c, items in per_code.items()},
        "note": "تصنيف معجمي مبسط للعناوين؛ تقدير عرضي لا يدخل في أي قرار رقمي ولا في نموذج التداول.",
    }
