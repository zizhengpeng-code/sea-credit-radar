"""RSS 并发抓取与正文提取。"""
from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import logging
import math
import re
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx

from .dedupe import clean_title, item_id, normalize_url
from .models import RawItem

log = logging.getLogger("radar.fetch")

UA = "Mozilla/5.0 (compatible; SEACreditRadar/0.1; +research)"
GN_BASE = "https://news.google.com/rss/search"


def feed_url(feed: dict[str, Any], window_days: int | None = None) -> str:
    """Google News 查询 -> RSS URL；window_days 用 when:Nd 限定时间窗。

    不加 when: 时 Google News 按相关度返回全部历史（实测最新一条可能是数月前），
    日常抓取必须带上，否则 48 小时窗口过滤后几乎为空。
    """
    if feed.get("url"):
        return feed["url"]
    q = feed["query"]
    if window_days:
        q += f" when:{window_days}d"
    return (f"{GN_BASE}?q={quote_plus(q)}&hl={feed.get('hl', 'en')}"
            f"&gl={feed.get('gl', 'US')}&ceid={feed.get('ceid', 'US:en')}")


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def parse_feed(content: bytes, feed: dict[str, Any], now: dt.datetime,
               window_hours: float) -> list[RawItem]:
    """把 RSS 字节流解析为 RawItem，丢弃窗口之外的条目。纯函数，便于测试。"""
    parsed = feedparser.parse(content)
    cutoff = now - dt.timedelta(hours=window_hours)
    items: list[RawItem] = []
    for e in parsed.entries:
        link = e.get("link")
        title = e.get("title")
        if not link or not title:
            continue
        pt = e.get("published_parsed") or e.get("updated_parsed")
        published = (dt.datetime.fromtimestamp(calendar.timegm(pt), dt.timezone.utc)
                     if pt else now)
        if published < cutoff:
            continue
        publisher = (e.get("source") or {}).get("title", "") if isinstance(e.get("source"), dict) else ""
        items.append(RawItem(
            id=item_id(link),
            url=normalize_url(link),
            title=clean_title(title, publisher),
            publisher=publisher,
            published=published,
            fetched_at=now,
            feed_name=feed.get("name", ""),
            country_hint=feed.get("country", "OTHER"),
            tier=int(feed.get("tier", 3)),
            snippet=_strip_html(e.get("summary", ""))[:500],
        ))
    return items


def is_junk_title(title: str, patterns: list[str]) -> bool:
    """site: 查询常返回站点页面（招聘、目录、登录页），按配置的正则黑名单在入库前剔除。"""
    return any(re.search(p, title, re.I) for p in patterns)


async def _fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, feed: dict[str, Any],
                     now: dt.datetime, window_hours: float) -> tuple[str, list[RawItem], str | None]:
    url = feed_url(feed, math.ceil(window_hours / 24))
    async with sem:
        try:
            r = await client.get(url)
            r.raise_for_status()
            return feed["name"], parse_feed(r.content, feed, now, window_hours), None
        except Exception as exc:  # 单个源失败不应拖垮整次运行
            return feed["name"], [], f"{type(exc).__name__}: {exc}"


async def fetch_all(cfg: dict[str, Any], window_hours: float | None = None,
                    window_days: int | None = None) -> tuple[list[RawItem], dict[str, Any]]:
    """并发抓取全部 feed。返回 (条目, 报告)；报告含每个源的条数与错误。"""
    fc = cfg.get("fetch", {})
    explicit = window_hours is not None or window_days is not None  # 回填：统一覆盖各 feed 的窗口
    default_hours = window_hours or (window_days * 24 if window_days else fc.get("window_hours", 48))
    now = dt.datetime.now(dt.timezone.utc)
    sem = asyncio.Semaphore(fc.get("concurrency", 8))
    async with httpx.AsyncClient(
        headers={"User-Agent": UA}, follow_redirects=True,
        timeout=fc.get("timeout_seconds", 20),
    ) as client:
        results = await asyncio.gather(*[
            _fetch_one(client, sem, f, now,
                       default_hours if explicit else f.get("window_hours", default_hours))
            for f in cfg["feeds"]
        ])
    items: list[RawItem] = []
    report: dict[str, Any] = {"per_feed": {}, "errors": {}}
    for name, its, err in results:
        report["per_feed"][name] = len(its)
        if err:
            report["errors"][name] = err
            log.warning("feed %s failed: %s", name, err)
        items.extend(its)
    junk = fc.get("title_blocklist", [])
    if junk:
        items = [i for i in items if not is_junk_title(i.title, junk)]
    # 不同 feed 可能返回同一 URL：先按 id 去一遍，保留等级更高的
    by_id: dict[str, RawItem] = {}
    for it in sorted(items, key=lambda i: i.tier):
        by_id.setdefault(it.id, it)
    return list(by_id.values()), report


# ---------- 正文提取（enrich 分支使用）----------
def resolve_google_news(url: str) -> str | None:
    """Google News 的 RSS 链接是加密跳转页，用 googlenewsdecoder 解出原文链接；失败返回 None。"""
    try:
        from googlenewsdecoder import gnewsdecoder

        d = gnewsdecoder(url, interval=0)
        return d.get("decoded_url") if d.get("success") else None
    except Exception as exc:
        log.info("gnews decode failed for %s: %s", url, exc)
        return None


def fetch_fulltext(url: str, timeout: float = 20, max_chars: int = 6000) -> str | None:
    """抓取页面并用 trafilatura 提取正文；任何一步失败返回 None（调用方回退到标题）。"""
    import trafilatura

    if "news.google.com" in url:
        url = resolve_google_news(url) or ""
        if not url:
            return None
    try:
        r = httpx.get(url, headers={"User-Agent": UA}, follow_redirects=True, timeout=timeout)
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        return text[:max_chars] if text else None
    except Exception as exc:
        log.info("fulltext failed for %s: %s", url, exc)
        return None
