"""流程编排：ingest（fetch + dedupe + 入库）与 process（extract -> 归并 -> enrich -> store）。"""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import local_tz
from .dedupe import dedupe_items, event_fingerprint, match_events
from .digest import Digest, build_digest, render_markdown
from .enrich import enrich_event, needs_enrich
from .extract import ItemExtraction, extract_all
from .fetch import fetch_all, fetch_fulltext
from .llm import LLMClient
from .models import Event, RawItem, Source, Update
from .notify import NotifyError, notify
from .store import Store

MAX_CANDIDATES_PER_BATCH = 8


def source_of(it: RawItem) -> Source:
    return Source(url=it.url, title=it.title, publisher=it.publisher,
                  published=it.published, tier=it.tier)


def _attach_source(store: Store, event_id: str, it: RawItem) -> None:
    """把条目及其被去重丢弃的近似重复报道，全部作为该事件的来源。"""
    store.add_source(event_id, source_of(it))
    store.set_item_event(it.id, event_id)
    for dup in store.duplicates_of(it.id):
        store.add_source(event_id, source_of(dup))
        store.set_item_event(dup.id, event_id)


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
    merged = 0
    for it, dup_of in dropped:
        store.mark_item(it.id, "duplicate", note=dup_of)
        eid = store.item_event_id(dup_of)  # 与已入事件的条目重复：作为该事件的新来源
        if eid:
            store.add_source(eid, source_of(it))
            store.set_item_event(it.id, eid)
            merged += 1

    return {
        "fetched": fetched,
        "new_url": len(new_items),
        "near_duplicates": len(dropped),
        "merged_into_events": merged,
        "to_process": len(kept),
        "feed_report": report,
    }


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
            limit: int | None = None, fetch_text: Callable[..., str | None] = fetch_fulltext,
            items: list[RawItem] | None = None) -> dict[str, Any]:
    """处理条目（默认库内 status=new）：抽取 -> 过滤 / 新建事件 / 并入既有事件（进展更新）-> enrich 分支。"""
    ec, dc = cfg.get("extract", {}), cfg.get("dedupe", {})
    cand_thr = dc.get("event_candidate_threshold", 0.3)
    lookback = dc.get("event_lookback_days", 30)
    bs = ec.get("batch_size", 8)
    if items is None:
        items = store.items_by_status("new", limit=limit or ec.get("max_items_per_run", 150))
    stats = {"processed": len(items), "filtered": 0, "events_new": 0, "updates": 0,
             "merged_sources": 0, "failed": 0, "enriched": 0}

    events = {e.id: e for e in store.recent_events(lookback, today)}
    titles = store.event_source_titles(events)
    touched: set[str] = set()

    def candidates_for(batch: list[RawItem]) -> list[dict]:
        ids: list[str] = []
        for it in batch:
            for eid, _ in match_events(it.title, titles, cand_thr)[:3]:
                if eid not in ids:
                    ids.append(eid)
        return [{"event_id": eid, "headline_cn": events[eid].headline_cn, "country": events[eid].country,
                 "last_updated": events[eid].last_updated.isoformat()}
                for eid in ids[:MAX_CANDIDATES_PER_BATCH]]

    for s in range(0, len(items), bs):
        batch = items[s: s + bs]
        cands = candidates_for(batch)  # 每批开始时算一次，含此前批次刚建的事件
        cand_ids = {c["event_id"] for c in cands}
        results = extract_all(llm, batch, len(batch), lambda _b: cands)
        for it, x, err in results:
            if x is None:
                store.mark_item(it.id, "failed", note=err)
                stats["failed"] += 1
                continue
            if not x.relevant:
                store.mark_item(it.id, "filtered")
                stats["filtered"] += 1
                continue

            eid = x.related_event_id if x.related_event_id in cand_ids else None  # 不信任模型编造的 id
            same_fingerprint = False
            if eid is None:
                fp = event_fingerprint(x.country, it.title)
                if fp in events:
                    eid, same_fingerprint = fp, True

            if eid is None:  # 全新事件
                ev = new_event(it, x, today)
                store.save_event(ev, title_key=it.title)
                events[ev.id] = ev
                titles[ev.id] = [it.title]
                touched.add(ev.id)
                stats["events_new"] += 1
                eid = ev.id
                _attach_source(store, eid, it)
            else:  # 既有事件的后续 / 重复报道
                ev = events[eid]
                _attach_source(store, eid, it)
                titles.setdefault(eid, []).append(it.title)
                stats["merged_sources"] += 1
                if x.adds_new_info and not same_fingerprint:
                    store.add_update(eid, Update(date=today, note=x.update_note or x.headline_cn,
                                                 source_url=it.url))
                    if x.effective_date and x.effective_date != ev.effective_date:
                        ev = ev.model_copy(update={"effective_date": x.effective_date, "last_updated": today})
                        store.save_event(ev)
                        events[eid] = ev
                    stats["updates"] += 1
                    touched.add(eid)
            store.mark_item(it.id, "linked", event_id=eid)

    # enrich 分支：只对监管类 / 低置信度事件抓正文做深度解读
    for eid in touched:
        ev = store.get_event(eid)
        if ev and needs_enrich(ev, cfg):
            richer = enrich_event(llm, ev, cfg, fetch_text)
            if richer:
                store.save_event(richer)
                stats["enriched"] += 1
    return stats


@dataclass
class RunResult:
    digest: Digest
    markdown: str
    ingest_stats: dict[str, Any]
    process_stats: dict[str, Any]
    digest_path: Path | None = None
    notified: str | None = None  # 实际使用的推送方式；None = 未推送
    notify_error: str | None = None


def digest_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["db_path"]).parent / "digests"


def save_digest(cfg: dict[str, Any], day: dt.date, markdown: str) -> Path:
    d = digest_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{day.isoformat()}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def run_pipeline(cfg: dict[str, Any], store: Store, llm: LLMClient, day: dt.date, *,
                 dry_run: bool = False, limit: int | None = None,
                 fetch_text: Callable[..., str | None] = fetch_fulltext) -> RunResult:
    """fetch -> dedupe -> extract -> enrich -> store -> digest -> notify。

    dry_run 只跳过推送；抓取、模型调用和入库照常进行（跨天的进展追踪依赖入库状态）。
    """
    ing = ingest(cfg, store)
    ing["expired"] = store.expire_stale(cfg.get("fetch", {}).get("expire_after_hours", 96))
    proc = process(cfg, store, llm, day, limit=limit, fetch_text=fetch_text)
    digest = build_digest(store, day, {**ing, **proc}, store.run_summary(llm.run_id), llm.model)
    md = render_markdown(digest)
    res = RunResult(digest, md, ing, proc, save_digest(cfg, day, md))
    if not dry_run:
        try:
            res.notified = notify(cfg, f"SEA Credit Radar {day.isoformat()}（新 {len(digest.new)} / 进展 {len(digest.updated)}）", md)
        except NotifyError as exc:  # 推送失败不能丢简报：已落盘，交由调用方提示
            res.notify_error = str(exc)
    return res


def backfill(cfg: dict[str, Any], store: Store, llm: LLMClient, days: int,
             fetch_text: Callable[..., str | None] = fetch_fulltext) -> dict[str, Any]:
    """回填历史：抓取最近 days 天，按发布日期从早到晚逐日处理，使 first_seen / 进展更新的日期贴近真实。"""
    ing = ingest(cfg, store, window_days=days)
    tz = local_tz(cfg)
    pending = store.items_by_status("new")
    by_day: dict[dt.date, list[RawItem]] = {}
    for it in pending:
        by_day.setdefault(it.published.astimezone(tz).date(), []).append(it)
    per_day = cfg.get("extract", {}).get("max_items_per_run", 150)
    totals = {"days": 0, "events_new": 0, "updates": 0, "filtered": 0, "failed": 0, "enriched": 0}
    for d in sorted(by_day):
        st = process(cfg, store, llm, d, items=by_day[d][:per_day], fetch_text=fetch_text)
        digest = build_digest(store, d, model=llm.model)
        save_digest(cfg, d, render_markdown(digest))
        totals["days"] += 1
        for k in ("events_new", "updates", "filtered", "failed", "enriched"):
            totals[k] += st[k]
    return {"ingest": ing, **totals}
