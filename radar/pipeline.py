"""流程编排：ingest（fetch + dedupe + 入库）与 process（extract -> store）。"""
from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from .dedupe import dedupe_items, event_fingerprint
from .extract import ItemExtraction, extract_all
from .fetch import fetch_all
from .llm import LLMClient
from .models import Event, RawItem, Source
from .store import Store


def ingest(cfg: dict[str, Any], store: Store, window_hours: float | None = None,
           window_days: int | None = None) -> dict[str, Any]:
    """抓取 -> URL 去重 -> 标题近似去重 -> 写入 raw_items。返回统计。"""
    items, report = asyncio.run(fetch_all(cfg, window_hours, window_days))
    fetched = len(items)

    # 1) URL 精确去重：只有真正新增的才继续
    new_items = store.add_raw_items(items)

    # 2) 标题近似去重：相对库内近期条目与本批内部
    lookback = cfg.get("dedupe", {}).get("event_lookback_days", 30)
    new_ids = {i.id for i in new_items}
    known = [(i, t) for i, t in store.known_titles(lookback) if i not in new_ids]
    kept, dropped = dedupe_items(new_items, known, cfg)
    for it, dup_of in dropped:
        store.mark_item(it.id, "duplicate", note=dup_of)

    return {
        "fetched": fetched,
        "new_url": len(new_items),
        "near_duplicates": len(dropped),
        "to_process": len(kept),
        "feed_report": report,
    }


def source_of(it: RawItem) -> Source:
    return Source(url=it.url, title=it.title, publisher=it.publisher,
                  published=it.published, tier=it.tier)


def new_event(it: RawItem, x: ItemExtraction, today: dt.date) -> Event:
    return Event(
        id=event_fingerprint(x.country or it.country_hint, it.title),
        first_seen=today, last_updated=today,
        country=x.country, category=x.category, entities=x.entities,
        headline_cn=x.headline_cn, why_it_matters=x.why_it_matters,
        impact_stage=x.impact_stage, effective_date=x.effective_date,
        confidence=x.confidence, sources=[source_of(it)],
    )


def process(cfg: dict[str, Any], store: Store, llm: LLMClient, today: dt.date,
            limit: int | None = None) -> dict[str, Any]:
    """处理库内 status=new 的条目：抽取 -> 过滤 / 建事件。"""
    ec = cfg.get("extract", {})
    limit = limit or ec.get("max_items_per_run", 150)
    items = store.items_by_status("new", limit=limit)
    stats = {"processed": len(items), "filtered": 0, "events_new": 0, "failed": 0}
    for it, x, err in extract_all(llm, items, ec.get("batch_size", 8)):
        if x is None:
            store.mark_item(it.id, "failed", note=err)
            stats["failed"] += 1
        elif not x.relevant:
            store.mark_item(it.id, "filtered")
            stats["filtered"] += 1
        else:
            ev = new_event(it, x, today)
            if store.get_event(ev.id) is None:
                store.save_event(ev, title_key=it.title)
                stats["events_new"] += 1
            store.add_source(ev.id, source_of(it))
            store.mark_item(it.id, "linked", event_id=ev.id)
    return stats
