"""数据模型。注意用 `import datetime as dt`，避免字段名 date 与类型 date 冲突。"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Country = Literal["ID", "PH", "SG", "VN", "OTHER"]
Category = Literal["regulation", "product", "funding", "market", "risk"]
Stage = Literal["acquisition", "underwriting", "limit", "pricing", "collection"]

STAGES: tuple[str, ...] = ("acquisition", "underwriting", "limit", "pricing", "collection")
CATEGORIES: tuple[str, ...] = ("regulation", "product", "funding", "market", "risk")
COUNTRIES: tuple[str, ...] = ("ID", "PH", "SG", "VN", "OTHER")


class RawItem(BaseModel):
    """从 RSS 抓到的原始条目。"""

    id: str
    url: str
    title: str  # 已去掉尾部「 - 媒体名」
    publisher: str = ""
    published: dt.datetime
    fetched_at: dt.datetime
    feed_name: str = ""
    country_hint: str = "OTHER"
    tier: int = 3
    snippet: str = ""


class Source(BaseModel):
    url: str
    title: str
    publisher: str = ""
    published: dt.datetime | None = None
    tier: int = 3


class Update(BaseModel):
    date: dt.date
    note: str
    source_url: str = ""


class Enrichment(BaseModel):
    """监管类 / 低置信度事件的深度解读（enrich 分支产出）。"""

    policy_points: list[str] = Field(default_factory=list)
    effective_date: dt.date | None = None
    business_impact: str = ""
    fulltext_used: bool = False


class Event(BaseModel):
    id: str  # 事件指纹
    first_seen: dt.date
    last_updated: dt.date
    country: Country
    category: Category
    entities: list[str] = Field(default_factory=list)
    headline_cn: str
    why_it_matters: str
    impact_stage: list[Stage] = Field(default_factory=list)
    effective_date: dt.date | None = None
    confidence: float = Field(ge=0, le=1)
    sources: list[Source] = Field(default_factory=list)
    updates: list[Update] = Field(default_factory=list)
    enrichment: Enrichment | None = None
