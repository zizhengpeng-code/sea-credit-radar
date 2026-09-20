import datetime as dt

from radar.models import Event, Source, Update
from radar.pipeline import ingest
from radar.store import Store

from conftest import make_item

D = dt.date(2026, 9, 21)


def _event(eid="e1") -> Event:
    return Event(id=eid, first_seen=D, last_updated=D, country="ID", category="regulation",
                 entities=["OJK"], headline_cn="OJK 收紧数字借贷监管", why_it_matters="影响准入",
                 impact_stage=["underwriting"], effective_date=dt.date(2026, 12, 1), confidence=0.8)


def test_add_raw_items_ignores_duplicate_url(store):
    it = make_item("A story", url="https://a.com/1")
    assert len(store.add_raw_items([it])) == 1
    assert store.add_raw_items([it]) == []
    assert store.count_items() == {"new": 1}


def test_mark_and_query_status(store):
    it = make_item("A story", url="https://a.com/1")
    store.add_raw_items([it])
    store.mark_item(it.id, "filtered", note="irrelevant")
    assert store.items_by_status("new") == []
    assert store.items_by_status("filtered")[0].id == it.id


def test_event_roundtrip_with_sources_and_updates(store):
    ev = _event()
    store.save_event(ev, title_key="ojk lending")
    store.add_source("e1", Source(url="https://a.com/1", title="T1", tier=1))
    assert store.add_source("e1", Source(url="https://a.com/1", title="T1")) is False  # 同 URL 不重复
    store.add_update("e1", Update(date=D + dt.timedelta(days=1), note="OJK 公布细则"))
    got = store.get_event("e1")
    assert got.headline_cn == ev.headline_cn
    assert got.effective_date == dt.date(2026, 12, 1)
    assert len(got.sources) == 1 and len(got.updates) == 1
    assert got.last_updated == D + dt.timedelta(days=1)


def test_recent_events_window(store):
    store.save_event(_event("e1"))
    assert [e.id for e in store.recent_events(7, D)] == ["e1"]
    assert store.recent_events(7, D + dt.timedelta(days=30)) == []


def test_log_run_and_summary(store):
    store.log_run(run_id="r1", stage="extract", model="m", prompt_tokens=100,
                  completion_tokens=50, cost_usd=0.001)
    store.log_run(run_id="r1", stage="enrich", model="m", prompt_tokens=200,
                  completion_tokens=80, cost_usd=0.002)
    s = store.run_summary("r1")
    assert s["calls"] == 2 and s["prompt_tokens"] == 300 and abs(s["cost_usd"] - 0.003) < 1e-9


def test_ingest_end_to_end(monkeypatch, cfg):
    import radar.pipeline as pl

    a = make_item("OJK tightens digital lending rules for fintech lenders", url="https://a.com/1", tier=1)
    b = make_item("OJK tightens rules on digital lending for fintech lenders", url="https://b.com/2", tier=3)
    c = make_item("Kredivo secures new debt funding from investors", url="https://c.com/3")

    async def fake_fetch_all(cfg, wh, wd):
        return [a, b, c], {"per_feed": {}, "errors": {}}

    monkeypatch.setattr(pl, "fetch_all", fake_fetch_all)
    store = Store(":memory:")
    res = ingest(cfg, store)
    assert res["fetched"] == 3 and res["new_url"] == 3
    assert res["near_duplicates"] == 1 and res["to_process"] == 2
    counts = store.count_items()
    assert counts == {"new": 2, "duplicate": 1}
    # 第二次运行：URL 全部已存在，不再新增
    res2 = ingest(cfg, store)
    assert res2["new_url"] == 0
    # 新一天另一家媒体再报道同一事件：与库内近似重复
    d = make_item("OJK tightens digital lending rules for fintech lender", url="https://d.com/4")

    async def fake2(cfg, wh, wd):
        return [d], {"per_feed": {}, "errors": {}}

    monkeypatch.setattr(pl, "fetch_all", fake2)
    res3 = ingest(cfg, store)
    assert res3["near_duplicates"] == 1 and res3["to_process"] == 0
