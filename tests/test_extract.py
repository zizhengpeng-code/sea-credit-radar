import datetime as dt
import json

import pytest

from radar.extract import (BatchExtraction, ItemExtraction, extract_all, extract_batch, fit,
                           normalize)
from radar.llm import LLMClient, LLMError
from radar.pipeline import process
from radar.prompts import EXTRACT_SYSTEM, build_extract_prompt

from conftest import FakeLLM, make_item

ENRICH_OK = {"policy_points": ["日利率上限 0.3%"], "effective_date": None,
             "business_impact": "压缩 BNPL 定价空间", "category": "regulation",
             "impact_stage": ["pricing"], "confidence": 0.9}

GOOD = {"idx": 0, "relevant": True, "country": "ID", "category": "regulation",
        "entities": ["OJK", "OJK", "Kredivo"], "headline_cn": "OJK 将 Paylater 日利率上限定为 0.3%",
        "why_it_matters": "直接压缩印尼 BNPL 产品定价空间", "impact_stage": ["pricing", "pricing"],
        "effective_date": "2026-12-01", "confidence": 0.85}


def test_relevant_requires_core_fields():
    with pytest.raises(ValueError, match="缺少字段"):
        ItemExtraction(idx=0, relevant=True, country="ID")


def test_irrelevant_needs_only_idx():
    x = ItemExtraction.model_validate({"idx": 3, "relevant": False})
    assert x.relevant is False and x.category is None


def test_invalid_enum_rejected():
    bad = dict(GOOD, category="gossip")
    with pytest.raises(ValueError):
        ItemExtraction.model_validate(bad)
    with pytest.raises(ValueError):
        ItemExtraction.model_validate(dict(GOOD, impact_stage=["marketing"]))
    with pytest.raises(ValueError):
        ItemExtraction.model_validate(dict(GOOD, confidence=1.5))


def test_normalize_dedupes_and_truncates():
    x = ItemExtraction.model_validate(dict(GOOD, headline_cn="字" * 45, entities=["A", "A", "B"]))
    x = normalize(x)
    assert len(x.headline_cn) == 30 and x.headline_cn.endswith("…")
    assert x.entities == ["A", "B"] and x.impact_stage == ["pricing"]


def test_fit():
    assert fit("abc", 5) == "abc"
    assert fit("abcdefgh", 5) == "abcd…"


def test_prompt_contains_items_and_candidates():
    items = [{"idx": 0, "title": "OJK news", "snippet": "", "publisher": "P", "published": "2026-09-20",
              "country_hint": "ID", "tier": 1}]
    cands = [{"event_id": "abc123", "headline_cn": "旧事件", "country": "ID", "last_updated": "2026-09-19"}]
    p = build_extract_prompt(items, cands)
    assert "event_id=abc123" in p and "标题: OJK news" in p and "idx 为 0" in p
    assert "已有事件" not in build_extract_prompt(items, None)


def test_system_prompt_forbids_vendor_features_and_fences():
    assert "禁止任何前言" in EXTRACT_SYSTEM and "Markdown 代码块" in EXTRACT_SYSTEM
    assert "tool" not in EXTRACT_SYSTEM.lower()


def test_extract_batch_maps_idx_and_ignores_bad_idx():
    items = [make_item("a story one"), make_item("a story two")]
    llm = FakeLLM([{"results": [GOOD, {"idx": 1, "relevant": False}, {"idx": 9, "relevant": False}]}])
    out = extract_batch(llm, items)
    assert set(out) == {0, 1} and out[0].relevant and not out[1].relevant


def test_extract_all_batches_and_records_failure():
    items = [make_item(f"story {i} about lending") for i in range(5)]
    llm = FakeLLM([
        {"results": [{"idx": i, "relevant": False} for i in range(2)]},
        LLMError("重试后仍无法通过校验"),
        {"results": [{"idx": 0, "relevant": False}]},
    ])
    res = extract_all(llm, items, batch_size=2)
    assert [r[1] is not None for r in res] == [True, True, False, False, True]
    assert "校验" in res[2][2]


def test_extract_all_marks_missing_item():
    items = [make_item("story a lending"), make_item("story b lending")]
    llm = FakeLLM([{"results": [{"idx": 0, "relevant": False}]}])
    res = extract_all(llm, items, batch_size=8)
    assert res[1][1] is None and "未返回" in res[1][2]


def test_process_creates_event_filters_and_fails(store, cfg):
    a = make_item("OJK caps paylater interest at 0.3% per day", url="https://a.com/1", tier=1)
    b = make_item("Volcano ash spreads across islands", url="https://a.com/2")
    c = make_item("Another item", url="https://a.com/3", hours_ago=5)
    store.add_raw_items([a, b, c])
    cfg["extract"] = {"batch_size": 8}
    llm = FakeLLM([{"results": [GOOD, {"idx": 1, "relevant": False}]},  # 第 3 条（idx 2）未返回
                   ENRICH_OK])  # GOOD 是监管类事件，会触发 enrich
    stats = process(cfg, store, llm, dt.date(2026, 9, 21), fetch_text=lambda url, **kw: None)
    assert stats == {"processed": 3, "filtered": 1, "events_new": 1, "updates": 0,
                     "merged_sources": 0, "failed": 1, "enriched": 1}
    ev = store.recent_events(1, dt.date(2026, 9, 21))[0]
    assert ev.category == "regulation" and ev.country == "ID"
    assert ev.effective_date == dt.date(2026, 12, 1) and len(ev.sources) == 1
    assert store.count_items() == {"linked": 1, "filtered": 1, "failed": 1}
    assert json.dumps(ev.impact_stage) == '["pricing"]'


# ---------- 生效日期校验 / 字数压缩 ----------
def test_sane_effective_date():
    from radar.models import sane_effective_date
    ref = dt.date(2026, 9, 21)
    assert sane_effective_date(None, ref) is None
    assert sane_effective_date(dt.date(2026, 12, 1), ref) == dt.date(2026, 12, 1)
    assert sane_effective_date(dt.date(2024, 6, 25), ref) is None   # 猜错年份（实测出现过）
    assert sane_effective_date(dt.date(2026, 7, 1), ref) == dt.date(2026, 7, 1)  # 近期已生效仍保留
    assert sane_effective_date(dt.date(2035, 1, 1), ref) is None


def test_extract_batch_drops_implausible_effective_date():
    items = [make_item("BSP bans SMS OTP for banks")]
    llm = FakeLLM([{"results": [dict(GOOD, effective_date="2024-06-25")]}])
    assert extract_batch(llm, items)[0].effective_date is None


def test_over_limit_text_is_shortened_by_model_not_truncated():
    items = [make_item("IFC approves loan to First Circle")]
    long = "IFC 批准向菲律宾 First Circle 提供最高一亿美元的融资额度用于中小企业放款"  # > 30 字
    llm = FakeLLM([{"results": [dict(GOOD, headline_cn=long)]},
                   {"items": [{"idx": 0, "text": "IFC 批准向 First Circle 提供1亿美元融资"}]}])
    x = extract_batch(llm, items)[0]
    assert x.headline_cn == "IFC 批准向 First Circle 提供1亿美元融资" and "…" not in x.headline_cn
    assert llm.stages == ["extract", "shorten"]


def test_shorten_failure_falls_back_to_truncation():
    items = [make_item("IFC approves loan to First Circle")]
    long = "字" * 45
    llm = FakeLLM([{"results": [dict(GOOD, headline_cn=long)]}, LLMError("boom")])
    x = extract_batch(llm, items)[0]
    assert len(x.headline_cn) == 30 and x.headline_cn.endswith("…")


def test_shorten_result_still_too_long_is_ignored_then_truncated():
    items = [make_item("IFC approves loan to First Circle")]
    long = "字" * 45
    llm = FakeLLM([{"results": [dict(GOOD, headline_cn=long)]}, {"items": [{"idx": 0, "text": "字" * 40}]}])
    assert len(extract_batch(llm, items)[0].headline_cn) == 30


def test_no_shorten_call_when_within_limits():
    llm = FakeLLM([{"results": [GOOD]}])
    extract_batch(llm, [make_item("OJK caps paylater")])
    assert llm.stages == ["extract"]
