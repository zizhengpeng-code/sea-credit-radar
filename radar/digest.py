"""每日简报：新事件与进展更新分区展示，按影响程度排序。"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .models import Event, Update
from .store import Store

CATEGORY_CN = {"regulation": "监管", "product": "产品", "funding": "融资", "market": "市场", "risk": "风险"}
STAGE_CN = {"acquisition": "获客", "underwriting": "准入", "limit": "额度", "pricing": "定价", "collection": "贷后"}
COUNTRY_CN = {"ID": "印尼", "PH": "菲律宾", "SG": "新加坡", "VN": "越南", "OTHER": "东南亚"}

# 影响程度打分：类别基础分 + 影响环节数 + 官方来源 + 多源印证，再按置信度折扣。分值只用于排序。
CATEGORY_WEIGHT = {"regulation": 5.0, "risk": 4.0, "product": 3.0, "funding": 3.0, "market": 2.0}


def impact_score(ev: Event) -> float:
    s = CATEGORY_WEIGHT.get(ev.category, 2.0)
    s += 0.5 * len(ev.impact_stage)
    if any(src.tier == 1 for src in ev.sources):
        s += 1.0
    s += 0.5 * min(max(len(ev.sources) - 1, 0), 3)
    return s * (0.5 + 0.5 * ev.confidence)


@dataclass
class DigestEntry:
    event: Event
    score: float
    prev_conclusion: str = ""  # 仅进展更新：上次结论
    today_updates: list[Update] = field(default_factory=list)


@dataclass
class Digest:
    day: dt.date
    new: list[DigestEntry]
    updated: list[DigestEntry]
    stats: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    model: str = ""

    @property
    def empty(self) -> bool:
        return not self.new and not self.updated


def build_digest(store: Store, day: dt.date, stats: dict[str, Any] | None = None,
                 cost: dict[str, Any] | None = None, model: str = "") -> Digest:
    new: list[DigestEntry] = []
    updated: list[DigestEntry] = []
    for ev in store.events_changed_on(day):
        today_updates = [u for u in ev.updates if u.date == day]
        if ev.first_seen == day:
            new.append(DigestEntry(ev, impact_score(ev)))
        elif today_updates:
            earlier = [u for u in ev.updates if u.date < day]
            prev = f"{earlier[-1].note}（{earlier[-1].date:%m-%d}）" if earlier \
                else f"{ev.headline_cn}（{ev.first_seen:%m-%d}）"
            updated.append(DigestEntry(ev, impact_score(ev), prev, today_updates))
    key = lambda e: -e.score  # noqa: E731
    return Digest(day, sorted(new, key=key), sorted(updated, key=key), stats or {}, cost or {}, model)


def _sources_line(ev: Event, n: int = 3) -> str:
    srcs = sorted(ev.sources, key=lambda s: (s.tier, s.published or dt.datetime.max.replace(tzinfo=dt.timezone.utc)))
    seen: set[str] = set()
    uniq = []
    for s in srcs:  # 同一媒体多条报道只显示一次
        name = s.publisher or "来源"
        if name not in seen:
            seen.add(name)
            uniq.append(s)
    parts = [f"[{s.publisher or '来源'}]({s.url})" for s in uniq[:n]]
    extra = len(uniq) - n
    return " · ".join(parts) + (f"（另 {extra} 条）" if extra > 0 else "")


def _entry_md(i: int, e: DigestEntry) -> list[str]:
    ev = e.event
    tag = f"{COUNTRY_CN.get(ev.country, ev.country)}·{CATEGORY_CN.get(ev.category, ev.category)}"
    low = " ⚠️低置信度" if ev.confidence < 0.6 else ""
    lines = [f"### {i}. 【{tag}】{ev.headline_cn}{low}"]
    if e.today_updates:
        lines.append(f"- **上次结论**：{e.prev_conclusion}")
        for u in e.today_updates:
            lines.append(f"- **本次进展**：{u.note}")
    lines.append(f"- **为什么重要**：{ev.why_it_matters}")
    meta = ["影响环节：" + ("、".join(STAGE_CN[s] for s in ev.impact_stage) if ev.impact_stage else "无明确环节")]
    if ev.effective_date:
        meta.append(f"生效：{ev.effective_date.isoformat()}")
    meta.append(f"置信度 {ev.confidence:.2f}")
    if ev.entities:
        meta.append("涉及：" + "、".join(ev.entities))
    lines.append("- " + " ｜ ".join(meta))
    if ev.enrichment:
        en = ev.enrichment
        lines.append("- **深度解读**" + ("" if en.fulltext_used else "（仅标题，未获取原文）") + "：")
        lines += [f"  - {p}" for p in en.policy_points]
        if en.business_impact:
            lines.append(f"  - 业务影响：{en.business_impact}")
    lines.append(f"- 来源：{_sources_line(ev)}")
    lines.append("")
    return lines


def render_markdown(d: Digest) -> str:
    lines = [f"# SEA Credit Radar · {d.day.isoformat()}", ""]
    summary = f"新事件 {len(d.new)} · 进展更新 {len(d.updated)}"
    if d.model:
        summary += f" · 模型 {d.model}"
    if d.cost:
        summary += (f" · tokens {d.cost.get('prompt_tokens', 0)}+{d.cost.get('completion_tokens', 0)}"
                    f" · 成本 ${d.cost.get('cost_usd', 0):.4f}")
    lines += [f"> {summary}", ""]
    if d.empty:
        lines.append("今日没有新的相关事件。")
        return "\n".join(lines) + "\n"
    if d.updated:
        lines += ["## 🔄 进展更新", ""]
        for i, e in enumerate(d.updated, 1):
            lines += _entry_md(i, e)
    if d.new:
        lines += ["## 🆕 新事件", ""]
        for i, e in enumerate(d.new, 1):
            lines += _entry_md(i, e)
    return "\n".join(lines) + "\n"
