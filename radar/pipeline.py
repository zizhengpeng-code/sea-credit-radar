"""流程编排。阶段一只有 ingest（fetch + dedupe + 入库）。"""
from __future__ import annotations

import asyncio
from typing import Any

from .dedupe import dedupe_items
from .fetch import fetch_all
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
