"""事件归并、进展追踪、enrich 分支的行为测试（模型调用与正文抓取全部 mock）。"""
import datetime as dt

import radar.pipeline as pl
from radar.enrich import enrich_event, needs_enrich
from radar.llm import LLMError
from radar.models import Enrichment, Event, Source
from radar.pipeline import ingest, process

from conftest import FakeLLM, make_item

D1 = dt.date(2026, 9, 20)
D2 = dt.date(2026, 9, 21)

NO_TEXT = lambda url, **kw: None  # noqa: E731


def ex(idx=0, **kw):
    base = {"idx": idx, "relevant": True, "country": "ID", "category": "product",
            "entities": ["Kredivo"], "headline_cn": "Kredivo 推出新产品", "why_it_matters": "影响获客",
            "impact_stage": ["acquisition"], "effective_date": None, "confidence": 0.9,
            "related_event_id": None, "adds_new_info": True, "update_note": None}
    base.update(kw)
    return base


def enrich_out(**kw):
    base = {"policy_points": ["要点一"], "effective_date": None, "business_impact": "影响 X",
            "category": "regulation", "impact_stage": ["pricing"], "confidence": 0.9}
    base.update(kw)
    return base


def seed_event(store, title="OJK sets new rules for digital lending platforms", **kw) -> Event:
    ev = Event(id="evt1", first_seen=D1, last_updated=D1, country="ID", category="product",
               entities=["OJK"], headline_cn="OJK 发布数字借贷新规", why_it_matters="影响准入",
               impact_stage=["underwriting"], confidence=0.9, **kw)
    store.save_event(ev, title_key=title)
    store.add_source("evt1", Source(url="https://old.com/1", title=title, tier=1))
    return ev


def process_one(store, cfg, item, llm, today=D2):
    store.add_raw_items([item])
    return process(cfg, store, llm, today, fetch_text=NO_TEXT)


# ---------- 进展追踪 ----------
def test_followup_becomes_update_and_keeps_previous_conclusion(store, cfg):
    seed_event(store)
    it = make_item("OJK deadline for digital lending platforms rules compliance", url="https://n.com/2")
    llm = FakeLLM([{"results": [ex(related_event_id="evt1", adds_new_info=True,
                                   update_note="OJK 公布合规截止日期")]}])
    stats = process_one(store, cfg, it, llm)
    assert stats["events_new"] == 0 and stats["updates"] == 1
    assert "event_id=evt1" in llm.prompts[0] and "OJK 发布数字借贷新规" in llm.prompts[0]
    ev = store.get_event("evt1")
    assert ev.headline_cn == "OJK 发布数字借贷新规"  # 上次结论保留
    assert [u.note for u in ev.updates] == ["OJK 公布合规截止日期"]
    assert ev.first_seen == D1 and ev.last_updated == D2
    assert len(ev.sources) == 2


def test_repeat_coverage_adds_source_without_update(store, cfg):
    seed_event(store)
    it = make_item("OJK sets rules for digital lending platforms says report", url="https://n.com/3")
    llm = FakeLLM([{"results": [ex(related_event_id="evt1", adds_new_info=False)]}])
    stats = process_one(store, cfg, it, llm)
    assert stats["updates"] == 0 and stats["merged_sources"] == 1
    ev = store.get_event("evt1")
    assert ev.updates == [] and len(ev.sources) == 2 and ev.last_updated == D1


def test_invented_event_id_is_ignored(store, cfg):
    seed_event(store)
    it = make_item("OJK deadline for digital lending platforms rules compliance", url="https://n.com/4")
    llm = FakeLLM([{"results": [ex(related_event_id="does-not-exist")]}])
    stats = process_one(store, cfg, it, llm)
    assert stats["events_new"] == 1 and stats["updates"] == 0


def test_effective_date_change_propagates_and_keeps_last_updated(store, cfg):
    seed_event(store)
    it = make_item("OJK deadline for digital lending platforms rules compliance", url="https://n.com/5")
    llm = FakeLLM([{"results": [ex(related_event_id="evt1", update_note="确定生效日",
                                   effective_date="2027-01-01")]}])
    process_one(store, cfg, it, llm)
    ev = store.get_event("evt1")
    assert ev.effective_date == dt.date(2027, 1, 1)
    assert ev.last_updated == D2  # 不能被旧的内存副本覆盖回 D1


def test_two_day_progress_flow(store, cfg):
    """第一天建事件，第二天出现后续报道 -> 同一事件出现一条进展更新。"""
    it1 = make_item("Bank Indonesia plans new paylater limits for lenders", url="https://n.com/a")
    llm1 = FakeLLM([{"results": [ex(headline_cn="印尼央行拟设 Paylater 额度上限")]}])
    process_one(store, cfg, it1, llm1, today=D1)
    eid = store.recent_events(1, D1)[0].id

    it2 = make_item("Bank Indonesia finalises paylater limits for lenders effective March", url="https://n.com/b")
    llm2 = FakeLLM([{"results": [ex(related_event_id=eid, update_note="细则敲定，3 月生效")]}])
    process_one(store, cfg, it2, llm2, today=D2)
    ev = store.get_event(eid)
    assert ev.first_seen == D1 and ev.last_updated == D2 and len(ev.updates) == 1
    assert [e.id for e in store.events_changed_on(D2)] == [eid]


# ---------- enrich 分支 ----------
def test_regulation_event_is_enriched_and_corrected(store, cfg):
    it = make_item("OJK caps paylater interest", url="https://n.com/r")
    llm = FakeLLM([{"results": [ex(category="regulation", confidence=0.9)]},
                   enrich_out(effective_date="2027-03-01", category="regulation", confidence=0.95)])
    store.add_raw_items([it])
    stats = process(cfg, store, llm, D2, fetch_text=lambda url, **kw: "全文内容")
    assert stats["enriched"] == 1 and llm.stages == ["extract", "enrich"]
    ev = store.recent_events(1, D2)[0]
    assert ev.enrichment.fulltext_used and ev.enrichment.policy_points == ["要点一"]
    assert ev.effective_date == dt.date(2027, 3, 1) and ev.confidence == 0.95


def test_high_confidence_non_regulation_is_not_enriched(store, cfg):
    it = make_item("Kredivo launches product", url="https://n.com/p")
    llm = FakeLLM([{"results": [ex(category="product", confidence=0.9)]}])
    stats = process_one(store, cfg, it, llm)
    assert stats["enriched"] == 0 and llm.stages == ["extract"]  # 只有一次模型调用


def test_low_confidence_event_is_enriched(store, cfg):
    it = make_item("Something vague about lending", url="https://n.com/l")
    llm = FakeLLM([{"results": [ex(category="market", confidence=0.4)]},
                   enrich_out(category="risk", impact_stage=["collection"], confidence=0.8)])
    stats = process_one(store, cfg, it, llm)
    assert stats["enriched"] == 1
    ev = store.recent_events(1, D2)[0]
    assert ev.category == "risk" and ev.impact_stage == ["collection"]  # 读过原文后修正归类


def test_enrich_without_fulltext_caps_confidence(store, cfg):
    ev = seed_event(store).model_copy(update={
        "category": "regulation", "sources": [Source(url="https://x.com", title="t", tier=1)]})
    llm = FakeLLM([enrich_out(confidence=0.95)])
    out = enrich_event(llm, ev, cfg, NO_TEXT)
    assert out.confidence == 0.6 and out.enrichment.fulltext_used is False
    assert "未能获取原文" in llm.prompts[0]


def test_needs_enrich_conditions(cfg):
    base = dict(id="x", first_seen=D1, last_updated=D1, country="ID", headline_cn="h",
                why_it_matters="w", impact_stage=[])
    assert needs_enrich(Event(category="regulation", confidence=0.95, **base), cfg)
    assert needs_enrich(Event(category="market", confidence=0.59, **base), cfg)
    assert not needs_enrich(Event(category="market", confidence=0.6, **base), cfg)
    assert not needs_enrich(Event(category="regulation", confidence=0.5,
                                  enrichment=Enrichment(), **base), cfg)  # 已解读过不重复


def test_enrich_failure_leaves_event_unchanged(store, cfg):
    ev = seed_event(store).model_copy(update={"category": "regulation"})
    assert enrich_event(FakeLLM([LLMError("boom")]), ev, cfg, NO_TEXT) is None


# ---------- 去重结果归并到事件来源 ----------
def test_ingest_attaches_near_duplicate_to_existing_event(monkeypatch, store, cfg):
    it = make_item("OJK tightens digital lending rules for fintech lenders", url="https://a.com/1", tier=1)
    store.add_raw_items([it])
    llm = FakeLLM([{"results": [ex(category="product", confidence=0.9)]}])
    process(cfg, store, llm, D1, fetch_text=NO_TEXT)
    eid = store.item_event_id(it.id)

    dup = make_item("OJK tightens rules on digital lending for fintech lenders", url="https://b.com/2")

    async def fake_fetch_all(cfg, wh, wd):
        return [dup], {"per_feed": {}, "errors": {}}

    monkeypatch.setattr(pl, "fetch_all", fake_fetch_all)
    res = ingest(cfg, store)
    assert res["near_duplicates"] == 1 and res["merged_into_events"] == 1
    assert {s.url for s in store.get_event(eid).sources} == {"https://a.com/1", "https://b.com/2"}


def test_batch_duplicates_become_sources_when_kept_item_links(monkeypatch, store, cfg):
    a = make_item("OJK tightens digital lending rules for fintech lenders", url="https://a.com/1", tier=1)
    b = make_item("OJK tightens rules on digital lending for fintech lenders", url="https://b.com/2")

    async def fake_fetch_all(cfg, wh, wd):
        return [a, b], {"per_feed": {}, "errors": {}}

    monkeypatch.setattr(pl, "fetch_all", fake_fetch_all)
    ingest(cfg, store)
    llm = FakeLLM([{"results": [ex(category="product", confidence=0.9)]}])
    stats = process(cfg, store, llm, D1, fetch_text=NO_TEXT)
    assert stats["processed"] == 1  # 只处理保留的那条
    ev = store.recent_events(1, D1)[0]
    assert len(ev.sources) == 2
