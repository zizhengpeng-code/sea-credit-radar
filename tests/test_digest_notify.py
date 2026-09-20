import datetime as dt

import httpx
import pytest

import radar.notify as nt
import radar.pipeline as pl
from radar.dedupe import event_fingerprint
from radar.digest import Digest, build_digest, impact_score, render_markdown
from radar.models import Enrichment, Event, Source, Update
from radar.notify import NotifyError, notify, send_email, send_serverchan
from radar.pipeline import backfill, run_pipeline

from conftest import FakeLLM, make_item

D1 = dt.date(2026, 9, 20)
D2 = dt.date(2026, 9, 21)
NO_TEXT = lambda url, **kw: None  # noqa: E731


def make_event(eid="e1", category="regulation", conf=0.9, first=D2, stages=("pricing",), tier=1, **kw) -> Event:
    return Event(id=eid, first_seen=first, last_updated=first, country="ID", category=category,
                 entities=["OJK"], headline_cn=f"标题-{eid}", why_it_matters="影响定价",
                 impact_stage=list(stages), confidence=conf,
                 sources=[Source(url=f"https://x.com/{eid}", title="t", publisher="Reuters", tier=tier)], **kw)


def save(store, ev: Event, updates=()):
    store.save_event(ev)
    for s in ev.sources:
        store.add_source(ev.id, s)
    for u in updates:
        store.add_update(ev.id, u)


# ---------- 排序 ----------
def test_impact_score_orders_regulation_above_market():
    reg = make_event("a", "regulation", stages=("pricing", "limit"))
    mkt = make_event("b", "market", stages=(), tier=3)
    assert impact_score(reg) > impact_score(mkt)


def test_lower_confidence_lowers_score():
    assert impact_score(make_event(conf=0.4)) < impact_score(make_event(conf=0.9))


def test_multi_source_raises_score():
    a = make_event()
    b = make_event().model_copy(update={"sources": a.sources + [
        Source(url="https://y.com", title="t2", tier=3), Source(url="https://z.com", title="t3", tier=3)]})
    assert impact_score(b) > impact_score(a)


# ---------- 分区 ----------
def test_digest_splits_new_and_updated_and_sorts(store):
    save(store, make_event("low", "market", stages=(), tier=3))
    save(store, make_event("high", "regulation"))
    save(store, make_event("old", first=D1), updates=[Update(date=D2, note="出台细则")])
    save(store, make_event("quiet", first=D1))  # 昨天的事件，今天没有进展 -> 不出现
    d = build_digest(store, D2)
    assert [e.event.id for e in d.new] == ["high", "low"]
    assert [e.event.id for e in d.updated] == ["old"]
    assert d.updated[0].prev_conclusion.startswith("标题-old")


def test_prev_conclusion_uses_last_earlier_update(store):
    save(store, make_event("e", first=D1), updates=[Update(date=D1, note="第一次进展"),
                                                      Update(date=D2, note="第二次进展")])
    d = build_digest(store, D2)
    assert "第一次进展" in d.updated[0].prev_conclusion
    assert [u.note for u in d.updated[0].today_updates] == ["第二次进展"]


def test_new_source_without_update_does_not_appear(store):
    save(store, make_event("e", first=D1))
    store.add_source("e", Source(url="https://another.com", title="转载", tier=3))
    store.touch_event("e", D2)
    assert build_digest(store, D2).empty


# ---------- 渲染 ----------
def test_render_sections_order_and_fields(store):
    enr = Enrichment(policy_points=["要点甲"], business_impact="收紧准入", fulltext_used=True)
    save(store, make_event("n", conf=0.5, effective_date=dt.date(2027, 1, 1), enrichment=enr))
    save(store, make_event("u", first=D1), updates=[Update(date=D2, note="本次新增内容")])
    md = render_markdown(build_digest(store, D2, cost={"prompt_tokens": 100, "completion_tokens": 20,
                                                        "cost_usd": 0.0012}, model="deepseek-chat"))
    assert md.index("## 🔄 进展更新") < md.index("## 🆕 新事件")
    for needle in ("上次结论", "本次进展**：本次新增内容", "⚠️低置信度", "生效：2027-01-01", "要点甲",
                   "业务影响：收紧准入", "影响环节：定价", "印尼·监管", "deepseek-chat", "$0.0012", "[Reuters]("):
        assert needle in md, needle


def test_render_marks_title_only_enrichment(store):
    save(store, make_event(enrichment=Enrichment(policy_points=["p"], fulltext_used=False)))
    assert "仅标题" in render_markdown(build_digest(store, D2))


def test_render_empty_digest(store):
    md = render_markdown(build_digest(store, D2))
    assert "今日没有新的相关事件" in md and "🆕" not in md


# ---------- 推送 ----------
def test_serverchan_posts_form(monkeypatch):
    seen = {}

    def fake_post(url, data=None, timeout=None):
        seen.update(url=url, data=data)
        return httpx.Response(200, json={"code": 0}, request=httpx.Request("POST", url))

    monkeypatch.setattr(nt.httpx, "post", fake_post)
    send_serverchan("标题", "正文", key="SCT123")
    assert seen["url"] == "https://sctapi.ftqq.com/SCT123.send"
    assert seen["data"] == {"title": "标题", "desp": "正文"}


def test_serverchan_truncates_long_body(monkeypatch):
    seen = {}
    monkeypatch.setattr(nt.httpx, "post", lambda url, data=None, timeout=None: (
        seen.update(data=data) or httpx.Response(200, json={"code": 0}, request=httpx.Request("POST", url))))
    send_serverchan("t", "字" * 40000, key="k")
    assert len(seen["data"]["desp"]) < 31000 and "已截断" in seen["data"]["desp"]


def test_serverchan_errors(monkeypatch):
    monkeypatch.delenv("SERVERCHAN_KEY", raising=False)
    with pytest.raises(NotifyError, match="SERVERCHAN_KEY"):
        send_serverchan("t", "b")
    monkeypatch.setattr(nt.httpx, "post", lambda *a, **k: httpx.Response(
        200, json={"code": 40001, "message": "bad key"}, request=httpx.Request("POST", "http://x")))
    with pytest.raises(NotifyError, match="bad key"):
        send_serverchan("t", "b", key="k")


def test_serverchan_error_does_not_leak_key(monkeypatch):
    def boom(url, **kw):
        raise httpx.ConnectError(f"cannot reach {url}")

    monkeypatch.setattr(nt.httpx, "post", boom)
    with pytest.raises(NotifyError) as ei:
        send_serverchan("t", "b", key="SECRETKEY")
    assert "SECRETKEY" not in str(ei.value)


def test_send_email(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, u, p):
            sent["login"] = (u, p)

        def sendmail(self, frm, to, msg):
            sent.update(frm=frm, to=to, msg=msg)

    for k, v in dict(SMTP_HOST="smtp.x.com", SMTP_USER="me@x.com", SMTP_PASSWORD="pw",
                     MAIL_TO="a@x.com, b@x.com", SMTP_PORT="465").items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(nt.smtplib, "SMTP_SSL", FakeSMTP)
    send_email("主题", "正文")
    assert sent["to"] == ["a@x.com", "b@x.com"] and sent["port"] == 465 and "Subject" in sent["msg"]


def test_email_requires_config(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(NotifyError):
        send_email("t", "b")


def test_notify_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(nt, "send_serverchan", lambda t, b: calls.append("sc"))
    monkeypatch.setattr(nt, "send_email", lambda t, b: calls.append("mail"))
    assert notify({"notify": {"method": "none"}}, "t", "b") == "none" and calls == []
    assert notify({"notify": {"method": "serverchan"}}, "t", "b") == "serverchan"
    assert notify({"notify": {"method": "email"}}, "t", "b") == "email"
    assert calls == ["sc", "mail"]
    with pytest.raises(NotifyError):
        notify({"notify": {"method": "carrier-pigeon"}}, "t", "b")


# ---------- run_pipeline / backfill ----------
def ex(idx=0, **kw):
    base = {"idx": idx, "relevant": True, "country": "ID", "category": "product", "entities": ["Kredivo"],
            "headline_cn": "Kredivo 推出新产品", "why_it_matters": "影响获客", "impact_stage": ["acquisition"],
            "effective_date": None, "confidence": 0.9, "related_event_id": None, "adds_new_info": True,
            "update_note": None}
    base.update(kw)
    return base


def patch_fetch(monkeypatch, items):
    async def fake(cfg, wh, wd):
        return items, {"per_feed": {}, "errors": {}}

    monkeypatch.setattr(pl, "fetch_all", fake)


@pytest.fixture
def cfg_db(cfg, tmp_path):
    cfg["db_path"] = str(tmp_path / "radar.db")
    return cfg


def test_dry_run_does_not_notify_but_saves_digest(monkeypatch, store, cfg_db):
    monkeypatch.setattr(pl, "notify", lambda *a: pytest.fail("dry-run 不应推送"))
    patch_fetch(monkeypatch, [make_item("Kredivo launches paylater product in Jakarta", url="https://a.com/1")])
    res = run_pipeline(cfg_db, store, FakeLLM([{"results": [ex()]}]), D2, dry_run=True, fetch_text=NO_TEXT)
    assert res.notified is None and res.digest_path.exists()
    assert "Kredivo 推出新产品" in res.digest_path.read_text(encoding="utf-8")


def test_live_run_notifies_and_survives_notify_failure(monkeypatch, store, cfg_db):
    patch_fetch(monkeypatch, [make_item("Kredivo launches paylater product in Jakarta", url="https://a.com/1")])
    got = {}
    monkeypatch.setattr(pl, "notify", lambda cfg, title, body: got.update(title=title, body=body) or "serverchan")
    res = run_pipeline(cfg_db, store, FakeLLM([{"results": [ex()]}]), D2, fetch_text=NO_TEXT)
    assert res.notified == "serverchan" and "新 1" in got["title"]

    def fail(*a):
        raise NotifyError("network down")

    monkeypatch.setattr(pl, "notify", fail)
    patch_fetch(monkeypatch, [make_item("Akulaku launches new limit product in Manila", url="https://a.com/2")])
    res = run_pipeline(cfg_db, store, FakeLLM([{"results": [ex(headline_cn="Akulaku 新额度产品")]}]), D2,
                       fetch_text=NO_TEXT)
    assert res.notify_error == "network down" and res.digest_path.exists()  # 简报不因推送失败而丢


def test_second_day_digest_has_progress_block(monkeypatch, store, cfg_db):
    """验收标准 3：连续运行两天后，第二天的简报出现「进展更新」区块。"""
    t1 = "Bank Indonesia plans new paylater limits for lenders"
    patch_fetch(monkeypatch, [make_item(t1, url="https://a.com/1")])
    r1 = run_pipeline(cfg_db, store, FakeLLM([{"results": [ex(headline_cn="印尼央行拟设 Paylater 额度上限")]}]),
                      D1, dry_run=True, fetch_text=NO_TEXT)
    assert "进展更新" not in r1.markdown.replace("进展更新 0", "")
    eid = event_fingerprint("ID", t1)

    t2 = "Bank Indonesia finalises paylater limits for lenders effective March"
    patch_fetch(monkeypatch, [make_item(t2, url="https://a.com/2")])
    r2 = run_pipeline(cfg_db, store, FakeLLM([{"results": [ex(related_event_id=eid, update_note="细则敲定，3 月生效")]}]),
                      D2, dry_run=True, fetch_text=NO_TEXT)
    assert "## 🔄 进展更新" in r2.markdown
    assert "上次结论" in r2.markdown and "印尼央行拟设 Paylater 额度上限" in r2.markdown
    assert "细则敲定，3 月生效" in r2.markdown
    assert len(r2.digest.updated) == 1 and len(r2.digest.new) == 0


def test_backfill_processes_days_in_order(monkeypatch, store, cfg_db):
    t1 = "Bank Indonesia plans new paylater limits for lenders"
    t2 = "Bank Indonesia finalises paylater limits for lenders effective March"
    items = [make_item(t2, url="https://a.com/2", hours_ago=1),    # 09-21 SGT
             make_item(t1, url="https://a.com/1", hours_ago=25)]   # 09-20 SGT（先处理）
    patch_fetch(monkeypatch, items)
    eid = event_fingerprint("ID", t1)
    llm = FakeLLM([{"results": [ex(headline_cn="印尼央行拟设额度上限")]},
                   {"results": [ex(related_event_id=eid, update_note="细则敲定")]}])
    res = backfill(cfg_db, store, llm, days=3, fetch_text=NO_TEXT)
    assert res["days"] == 2 and res["events_new"] == 1 and res["updates"] == 1
    ev = store.get_event(eid)
    assert ev.first_seen == D1 and ev.last_updated == D2 and len(ev.updates) == 1
    assert (pl.digest_dir(cfg_db) / "2026-09-21.md").exists()


def test_sources_line_dedupes_same_publisher(store):
    ev = make_event().model_copy(update={"sources": [
        Source(url="https://p.com/1", title="a", publisher="Philstar.com", tier=3),
        Source(url="https://p.com/2", title="b", publisher="Philstar.com", tier=3),
        Source(url="https://r.com/1", title="c", publisher="Reuters", tier=3)]})
    save(store, ev)
    md = render_markdown(build_digest(store, D2))
    assert md.count("[Philstar.com]") == 1 and "[Reuters]" in md
