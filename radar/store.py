"""SQLite 持久化：原始条目、事件（含来源与进展）、模型调用记录。"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Enrichment, Event, RawItem, Source, Update

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_items (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    publisher    TEXT DEFAULT '',
    published    TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    feed_name    TEXT DEFAULT '',
    country_hint TEXT DEFAULT 'OTHER',
    tier         INTEGER DEFAULT 3,
    snippet      TEXT DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'new',   -- new | linked | filtered | failed
    event_id     TEXT,
    note         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_items(status);

CREATE TABLE IF NOT EXISTS events (
    id             TEXT PRIMARY KEY,
    first_seen     TEXT NOT NULL,
    last_updated   TEXT NOT NULL,
    country        TEXT NOT NULL,
    category       TEXT NOT NULL,
    entities       TEXT NOT NULL DEFAULT '[]',
    headline_cn    TEXT NOT NULL,
    why_it_matters TEXT NOT NULL,
    impact_stage   TEXT NOT NULL DEFAULT '[]',
    effective_date TEXT,
    confidence     REAL NOT NULL,
    enrichment     TEXT,
    title_key      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS event_sources (
    event_id  TEXT NOT NULL REFERENCES events(id),
    url       TEXT NOT NULL UNIQUE,
    title     TEXT NOT NULL,
    publisher TEXT DEFAULT '',
    published TEXT,
    tier      INTEGER DEFAULT 3
);
CREATE INDEX IF NOT EXISTS idx_src_event ON event_sources(event_id);

CREATE TABLE IF NOT EXISTS updates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL REFERENCES events(id),
    date       TEXT NOT NULL,
    note       TEXT NOT NULL,
    source_url TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_upd_event ON updates(event_id);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    run_id            TEXT NOT NULL,
    stage             TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_version    TEXT DEFAULT '',
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd          REAL DEFAULT 0,
    latency_s         REAL DEFAULT 0,
    ok                INTEGER DEFAULT 1,
    note              TEXT DEFAULT ''
);
"""


def _iso(v: dt.date | dt.datetime | None) -> str | None:
    return v.isoformat() if v else None


class Store:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- raw items ----------
    def add_raw_items(self, items: Iterable[RawItem]) -> list[RawItem]:
        """写入原始条目，返回真正新增的那些（按 id 或 url 已存在则忽略）。"""
        added: list[RawItem] = []
        for it in items:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO raw_items
                   (id,url,title,publisher,published,fetched_at,feed_name,country_hint,tier,snippet)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (it.id, it.url, it.title, it.publisher, it.published.isoformat(),
                 it.fetched_at.isoformat(), it.feed_name, it.country_hint, it.tier, it.snippet),
            )
            if cur.rowcount:
                added.append(it)
        self.conn.commit()
        return added

    @staticmethod
    def _raw(row: sqlite3.Row) -> RawItem:
        return RawItem(
            id=row["id"], url=row["url"], title=row["title"], publisher=row["publisher"],
            published=dt.datetime.fromisoformat(row["published"]),
            fetched_at=dt.datetime.fromisoformat(row["fetched_at"]),
            feed_name=row["feed_name"], country_hint=row["country_hint"],
            tier=row["tier"], snippet=row["snippet"],
        )

    def items_by_status(self, status: str, limit: int | None = None) -> list[RawItem]:
        sql = "SELECT * FROM raw_items WHERE status=? ORDER BY tier ASC, published DESC"
        params: list = [status]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._raw(r) for r in self.conn.execute(sql, params)]

    def get_raw(self, item_id: str) -> RawItem | None:
        row = self.conn.execute("SELECT * FROM raw_items WHERE id=?", (item_id,)).fetchone()
        return self._raw(row) if row else None

    def mark_item(self, item_id: str, status: str, event_id: str | None = None, note: str = "") -> None:
        self.conn.execute(
            "UPDATE raw_items SET status=?, event_id=?, note=? WHERE id=?",
            (status, event_id, note, item_id),
        )
        self.conn.commit()

    def known_titles(self, days: int) -> list[tuple[str, str]]:
        """近 days 天内已入库条目的 (id, title)，用于跨批次去重。"""
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT id, title FROM raw_items WHERE fetched_at >= ?", (since,)
        ).fetchall()
        return [(r["id"], r["title"]) for r in rows]

    def count_items(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) c FROM raw_items GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    # ---------- events ----------
    def _event_from_row(self, row: sqlite3.Row) -> Event:
        eid = row["id"]
        sources = [
            Source(url=s["url"], title=s["title"], publisher=s["publisher"],
                   published=dt.datetime.fromisoformat(s["published"]) if s["published"] else None,
                   tier=s["tier"])
            for s in self.conn.execute(
                "SELECT * FROM event_sources WHERE event_id=? ORDER BY published", (eid,))
        ]
        updates = [
            Update(date=dt.date.fromisoformat(u["date"]), note=u["note"], source_url=u["source_url"])
            for u in self.conn.execute("SELECT * FROM updates WHERE event_id=? ORDER BY id", (eid,))
        ]
        return Event(
            id=eid,
            first_seen=dt.date.fromisoformat(row["first_seen"]),
            last_updated=dt.date.fromisoformat(row["last_updated"]),
            country=row["country"], category=row["category"],
            entities=json.loads(row["entities"]),
            headline_cn=row["headline_cn"], why_it_matters=row["why_it_matters"],
            impact_stage=json.loads(row["impact_stage"]),
            effective_date=dt.date.fromisoformat(row["effective_date"]) if row["effective_date"] else None,
            confidence=row["confidence"], sources=sources, updates=updates,
            enrichment=Enrichment.model_validate_json(row["enrichment"]) if row["enrichment"] else None,
        )

    def get_event(self, event_id: str) -> Event | None:
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._event_from_row(row) if row else None

    def recent_events(self, days: int, today: dt.date) -> list[Event]:
        since = (today - dt.timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT * FROM events WHERE last_updated >= ? ORDER BY last_updated DESC", (since,)
        ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def events_changed_on(self, day: dt.date) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE last_updated = ? OR first_seen = ?",
            (day.isoformat(), day.isoformat()),
        ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def save_event(self, ev: Event, title_key: str = "") -> None:
        """新建或覆盖事件主体字段（来源与进展另行追加）。"""
        self.conn.execute(
            """INSERT INTO events
               (id,first_seen,last_updated,country,category,entities,headline_cn,why_it_matters,
                impact_stage,effective_date,confidence,enrichment,title_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 last_updated=excluded.last_updated, country=excluded.country,
                 category=excluded.category, entities=excluded.entities,
                 headline_cn=excluded.headline_cn, why_it_matters=excluded.why_it_matters,
                 impact_stage=excluded.impact_stage, effective_date=excluded.effective_date,
                 confidence=excluded.confidence, enrichment=excluded.enrichment""",
            (ev.id, ev.first_seen.isoformat(), ev.last_updated.isoformat(), ev.country, ev.category,
             json.dumps(ev.entities, ensure_ascii=False), ev.headline_cn, ev.why_it_matters,
             json.dumps(ev.impact_stage), _iso(ev.effective_date), ev.confidence,
             ev.enrichment.model_dump_json() if ev.enrichment else None, title_key),
        )
        self.conn.commit()

    def add_source(self, event_id: str, src: Source) -> bool:
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO event_sources (event_id,url,title,publisher,published,tier)
               VALUES (?,?,?,?,?,?)""",
            (event_id, src.url, src.title, src.publisher, _iso(src.published), src.tier),
        )
        self.conn.commit()
        return bool(cur.rowcount)

    def add_update(self, event_id: str, upd: Update) -> None:
        self.conn.execute(
            "INSERT INTO updates (event_id,date,note,source_url) VALUES (?,?,?,?)",
            (event_id, upd.date.isoformat(), upd.note, upd.source_url),
        )
        self.conn.execute("UPDATE events SET last_updated=? WHERE id=?", (upd.date.isoformat(), event_id))
        self.conn.commit()

    def touch_event(self, event_id: str, day: dt.date) -> None:
        self.conn.execute("UPDATE events SET last_updated=? WHERE id=?", (day.isoformat(), event_id))
        self.conn.commit()

    def event_source_titles(self, event_ids: Iterable[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for eid in event_ids:
            out[eid] = [r["title"] for r in self.conn.execute(
                "SELECT title FROM event_sources WHERE event_id=?", (eid,))]
        return out

    # ---------- runs（模型调用与成本）----------
    def log_run(self, *, run_id: str, stage: str, model: str, prompt_version: str = "",
                prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0,
                latency_s: float = 0.0, ok: bool = True, note: str = "") -> None:
        self.conn.execute(
            """INSERT INTO runs (ts,run_id,stage,model,prompt_version,prompt_tokens,
               completion_tokens,cost_usd,latency_s,ok,note) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (dt.datetime.now(dt.timezone.utc).isoformat(), run_id, stage, model, prompt_version,
             prompt_tokens, completion_tokens, cost_usd, latency_s, int(ok), note),
        )
        self.conn.commit()

    def run_summary(self, run_id: str) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) calls, COALESCE(SUM(prompt_tokens),0) pt,
                      COALESCE(SUM(completion_tokens),0) ct, COALESCE(SUM(cost_usd),0) cost
               FROM runs WHERE run_id=?""", (run_id,)).fetchone()
        return {"calls": row["calls"], "prompt_tokens": row["pt"],
                "completion_tokens": row["ct"], "cost_usd": row["cost"]}
