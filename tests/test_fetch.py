import datetime as dt

import httpx


from radar import fetch
from radar.fetch import feed_url, is_junk_title, parse_feed

from conftest import NOW

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item>
  <title>OJK tightens digital lending rules - Jakarta Post</title>
  <link>https://news.google.com/rss/articles/AAA?oc=5</link>
  <pubDate>Sun, 20 Sep 2026 20:00:00 GMT</pubDate>
  <description>&lt;a href="x"&gt;OJK tightens&lt;/a&gt; rules</description>
  <source url="https://jakartapost.com">Jakarta Post</source>
</item>
<item>
  <title>Old story - Reuters</title>
  <link>https://news.google.com/rss/articles/BBB?oc=5</link>
  <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
  <source url="https://reuters.com">Reuters</source>
</item>
<item>
  <title>No link item</title>
</item>
</channel></rss>""".encode()


def test_parse_feed_filters_window_and_cleans():
    feed = {"name": "F", "country": "ID", "tier": 2}
    items = parse_feed(RSS, feed, NOW, window_hours=48)
    assert len(items) == 1
    it = items[0]
    assert it.title == "OJK tightens digital lending rules"
    assert it.publisher == "Jakarta Post"
    assert it.country_hint == "ID" and it.tier == 2
    assert it.url == "https://news.google.com/rss/articles/AAA"
    assert "<a" not in it.snippet
    assert it.published == dt.datetime(2026, 9, 20, 20, 0, tzinfo=dt.timezone.utc)


def test_parse_feed_wider_window_keeps_old():
    items = parse_feed(RSS, {"name": "F"}, NOW, window_hours=24 * 365)
    assert len(items) == 2


def test_feed_url_builds_google_news_query():
    u = feed_url({"query": "BNPL Philippines BSP", "hl": "en", "gl": "PH", "ceid": "PH:en"})
    assert u.startswith("https://news.google.com/rss/search?q=BNPL+Philippines+BSP")
    assert "gl=PH" in u and "ceid=PH:en" in u
    assert "when%3A7d" in feed_url({"query": "x"}, window_days=7) or "when:7d" in feed_url({"query": "x"}, window_days=7)


def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(fetch.httpx, "AsyncClient", factory)


def test_fetch_all_isolates_feed_failures(monkeypatch, cfg):
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        if "bsp.gov.ph" in str(request.url) or "bsp.gov.ph" in request.url.query.decode():
            return httpx.Response(500)
        return httpx.Response(200, content=RSS)

    _patch_client(monkeypatch, handler)
    items, report = asyncio.run(fetch.fetch_all(cfg, window_hours=24 * 365))
    assert len(items) == 2  # F1 成功（含旧条目，因窗口放宽）
    assert report["per_feed"]["F1"] == 2
    assert "F2" in report["errors"]


def test_fetch_all_merges_same_url_keeping_best_tier(monkeypatch, cfg):
    import asyncio

    _patch_client(monkeypatch, lambda req: httpx.Response(200, content=RSS))
    items, _ = asyncio.run(fetch.fetch_all(cfg, window_hours=48))
    assert len(items) == 1
    assert items[0].tier == 1  # F2(tier1) 与 F1(tier3) 同 URL，保留 tier 1


def test_is_junk_title():
    pats = ["job details", "^see all", "directory"]
    assert is_junk_title("Bank Officer IV Job Details", pats)
    assert is_junk_title("Financial Institutions Directory", pats)
    assert not is_junk_title("OJK tightens digital lending rules", pats)
