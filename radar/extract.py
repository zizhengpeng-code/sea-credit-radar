"""大模型结构化抽取：新闻条目 -> 相关性判断 + 事件字段。"""
from __future__ import annotations

import datetime as dt
import logging

from pydantic import BaseModel, Field, model_validator

from .llm import LLMClient, LLMError
from .models import Category, Country, RawItem, Stage, sane_effective_date
from .prompts import (EXTRACT_SYSTEM, EXTRACT_VERSION, SHORTEN_SYSTEM, SHORTEN_VERSION,
                      build_extract_prompt, build_shorten_prompt)

log = logging.getLogger("radar.extract")

HEADLINE_MAX, WHY_MAX = 30, 40  # 业务约束；schema 放宽到 2 倍，超出 30/40 的按字数截断，避免整批重试


class ItemExtraction(BaseModel):
    idx: int
    relevant: bool
    country: Country | None = None
    category: Category | None = None
    entities: list[str] = Field(default_factory=list)
    headline_cn: str = Field(default="", max_length=HEADLINE_MAX * 2)
    why_it_matters: str = Field(default="", max_length=WHY_MAX * 2)
    impact_stage: list[Stage] = Field(default_factory=list)
    effective_date: dt.date | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    related_event_id: str | None = None
    adds_new_info: bool = True
    update_note: str | None = None

    @model_validator(mode="after")
    def _require_fields_when_relevant(self) -> "ItemExtraction":
        if self.relevant:
            missing = [f for f in ("country", "category", "confidence")
                       if getattr(self, f) is None]
            if not self.headline_cn.strip():
                missing.append("headline_cn")
            if not self.why_it_matters.strip():
                missing.append("why_it_matters")
            if missing:
                raise ValueError(f"idx={self.idx} relevant=true 但缺少字段: {', '.join(missing)}")
        return self


class BatchExtraction(BaseModel):
    results: list[ItemExtraction]


def fit(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def normalize(x: ItemExtraction, ref: dt.date | None = None) -> ItemExtraction:
    """字数兜底截断、实体数量上限、去重、生效日期合理性校验（ref 为新闻发布日期）。"""
    if ref is not None:
        x.effective_date = sane_effective_date(x.effective_date, ref)
    x.headline_cn = fit(x.headline_cn, HEADLINE_MAX)
    x.why_it_matters = fit(x.why_it_matters, WHY_MAX)
    if x.update_note:
        x.update_note = fit(x.update_note, WHY_MAX)
    seen: list[str] = []
    for e in x.entities:
        if e and e not in seen:
            seen.append(e)
    x.entities = seen[:5]
    x.impact_stage = list(dict.fromkeys(x.impact_stage))
    return x


class ShortenItem(BaseModel):
    idx: int
    text: str = Field(min_length=1, max_length=200)


class ShortenBatch(BaseModel):
    items: list[ShortenItem]


_LIMITS = {"headline_cn": HEADLINE_MAX, "why_it_matters": WHY_MAX, "update_note": WHY_MAX}


def shorten_over_limit(llm: LLMClient, xs: list[ItemExtraction]) -> int:
    """对超出字数上限的字段，用一次小调用让模型自己压缩（比硬截断读起来完整）。返回成功压缩的字段数。

    失败不抛异常：保持原文，交给 normalize 的截断兜底。
    """
    jobs, targets = [], []
    for x in xs:
        for f, lim in _LIMITS.items():
            v = getattr(x, f)
            if v and len(v) > lim:
                jobs.append({"idx": len(jobs), "text": v, "max_chars": lim})
                targets.append((x, f, lim))
    if not jobs:
        return 0
    try:
        data = llm.complete(build_shorten_prompt(jobs), SHORTEN_SYSTEM, ShortenBatch,
                            stage="shorten", prompt_version=SHORTEN_VERSION)
    except LLMError as exc:
        log.warning("压缩失败，退回截断：%s", exc)
        return 0
    done = 0
    for r in ShortenBatch.model_validate(data).items:
        if 0 <= r.idx < len(targets):
            x, f, lim = targets[r.idx]
            if len(r.text.strip()) <= lim:
                setattr(x, f, r.text.strip())
                done += 1
    return done


def _payload(items: list[RawItem]) -> list[dict]:
    return [{"idx": i, "title": it.title, "snippet": it.snippet, "publisher": it.publisher,
             "published": it.published.strftime("%Y-%m-%d"), "country_hint": it.country_hint,
             "tier": it.tier} for i, it in enumerate(items)]


def extract_batch(llm: LLMClient, items: list[RawItem], candidates: list[dict] | None = None
                  ) -> dict[int, ItemExtraction]:
    """抽取一批条目，返回 {批内序号: 结果}。模型漏掉的条目不在返回值里（由调用方处理）。"""
    prompt = build_extract_prompt(_payload(items), candidates)
    data = llm.complete(prompt, EXTRACT_SYSTEM, BatchExtraction, stage="extract",
                        prompt_version=EXTRACT_VERSION)
    out: dict[int, ItemExtraction] = {}
    for r in BatchExtraction.model_validate(data).results:
        if 0 <= r.idx < len(items) and r.idx not in out:
            out[r.idx] = r
    shorten_over_limit(llm, [x for x in out.values() if x.relevant])
    return {i: normalize(x, items[i].published.date()) for i, x in out.items()}


def extract_all(llm: LLMClient, items: list[RawItem], batch_size: int = 8,
                candidates_for: "callable | None" = None
                ) -> list[tuple[RawItem, ItemExtraction | None, str]]:
    """分批抽取。返回 [(条目, 结果或 None, 失败原因)]。整批失败（重试后仍非法）则该批全部记为 None。"""
    results: list[tuple[RawItem, ItemExtraction | None, str]] = []
    for s in range(0, len(items), batch_size):
        batch = items[s: s + batch_size]
        cands = candidates_for(batch) if candidates_for else None
        try:
            got = extract_batch(llm, batch, cands)
        except LLMError as exc:
            log.warning("批次 %d-%d 失败，跳过：%s", s, s + len(batch), exc)
            results.extend((it, None, str(exc)[:200]) for it in batch)
            continue
        for i, it in enumerate(batch):
            r = got.get(i)
            results.append((it, r, "" if r else "模型未返回该条"))
    return results
