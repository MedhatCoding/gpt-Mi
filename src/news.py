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


def sentiment_summary(news: dict) -> dict:
    """متوسط درجات العناوين للسوق ولكل سهم من المختارات."""
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
