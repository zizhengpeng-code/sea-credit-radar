"""监管类 / 低置信度事件的深度解读分支：抓正文 -> 模型解读 -> 修正事件字段。

只有 category == regulation 或 confidence < 阈值 的事件才走这一步；其余事件只保留一句话摘要，控制成本。
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable

from pydantic import BaseModel, Field

from .fetch import fetch_fulltext
from .llm import LLMClient, LLMError
from .models import Category, Enrichment, Event, Stage, sane_effective_date
from .prompts import ENRICH_SYSTEM, ENRICH_VERSION, build_enrich_prompt

log = logging.getLogger("radar.enrich")

NO_FULLTEXT_CONF_CAP = 0.6  # 没读到原文时，置信度上限


class EnrichOut(BaseModel):
    policy_points: list[str] = Field(default_factory=list, max_length=8)
    effective_date: dt.date | None = None
    business_impact: str = Field(default="", max_length=400)
    category: Category
    impact_stage: list[Stage] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def needs_enrich(ev: Event, cfg: dict[str, Any]) -> bool:
    """分支条件：监管类，或置信度低；已解读过的不重复。"""
    if ev.enrichment is not None:
        return False
    return ev.category == "regulation" or ev.confidence < cfg.get("enrich", {}).get("confidence_below", 0.6)


def enrich_event(llm: LLMClient, ev: Event, cfg: dict[str, Any],
                 fetch: Callable[..., str | None] = fetch_fulltext) -> Event | None:
    """返回修正后的事件（含 enrichment）；模型调用失败返回 None，事件保持原样。"""
    max_chars = cfg.get("enrich", {}).get("max_fulltext_chars", 6000)
    text = None
    for src in sorted(ev.sources, key=lambda s: s.tier)[:2]:  # 最多尝试两个来源
        text = fetch(src.url, max_chars=max_chars)
        if text:
            break
    ref = max([s.published.date() for s in ev.sources if s.published] or [ev.last_updated])
    prompt = build_enrich_prompt(
        {"today": ref.isoformat(), "country": ev.country, "category": ev.category, "impact_stage": ev.impact_stage,
         "confidence": ev.confidence, "headline_cn": ev.headline_cn,
         "titles": [s.title for s in ev.sources]}, text)
    try:
        out = EnrichOut.model_validate(
            llm.complete(prompt, ENRICH_SYSTEM, EnrichOut, stage="enrich", prompt_version=ENRICH_VERSION))
    except LLMError as exc:
        log.warning("enrich 失败 %s：%s", ev.id, exc)
        return None
    conf = out.confidence if text else min(out.confidence, NO_FULLTEXT_CONF_CAP)
    eff = sane_effective_date(out.effective_date, ref)
    updated = ev.model_copy(update={
        "category": out.category,
        "impact_stage": list(dict.fromkeys(out.impact_stage)),
        "confidence": conf,
        "effective_date": eff or ev.effective_date,
        "enrichment": Enrichment(
            policy_points=[p[:80] for p in out.policy_points[:5]],
            effective_date=eff,
            business_impact=out.business_impact[:200],
            fulltext_used=bool(text),
        ),
    })
    return updated
