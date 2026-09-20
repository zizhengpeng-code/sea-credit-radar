"""去重与事件归并。

三层：
1. URL 精确去重（规范化后；入库时靠 UNIQUE 约束兜底）
2. 标题近似去重：SimHash 汉明距离 或 词集 Jaccard，任一命中即视为同一报道
3. 事件匹配：新条目与既有事件的所有来源标题比对，给出 (事件, 相似度) 候选，
   由调用方按阈值决定「直接并入」还是「交给模型判断是否为进展更新」
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import RawItem

_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
             "fbclid", "gclid", "ref", "oc", "cmpid"}

_STOP = frozenset("""a an the of to in on for and or with by from at as is are was were be been
its it this that these those will would may could should amid after over into new says say said
than more up out about vs via""".split())


# ---------- URL / 标题规范化 ----------
def normalize_url(url: str) -> str:
    """去掉 fragment、追踪参数，主机名小写，去掉末尾斜杠。Google News 文章链接整体去掉 query。"""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host == "news.google.com":
        query = ""
    else:
        query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def item_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode()).hexdigest()[:16]


def clean_title(title: str, publisher: str = "") -> str:
    """Google News 标题形如「Headline - Publisher」，去掉尾部媒体名。"""
    t = re.sub(r"\s+", " ", title).strip()
    if publisher and t.lower().endswith(" - " + publisher.lower()):
        return t[: -(len(publisher) + 3)].rstrip()
    m = re.match(r"^(.*\S)\s+[-–|]\s+([^-–|]{2,40})$", t)  # 没有 publisher 信息时的兜底
    if m and len(m.group(1)) > 15:
        return m.group(1)
    return t


def tokenize(title: str) -> list[str]:
    toks = re.findall(r"\w+", title.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # 极简复数还原：lenders -> lender
        out.append(t)
    return out


# ---------- 相似度 ----------
def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def simhash(tokens: list[str]) -> int:
    """64 位 SimHash，特征 = 词 + 相邻词二元组。"""
    feats = list(tokens) + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    if not feats:
        return 0
    v = [0] * 64
    for f in feats:
        h = _h64(f)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(64) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def title_similarity(t1: str, t2: str) -> float:
    return jaccard(tokenize(t1), tokenize(t2))


# 月份缩写前必须是词边界，且后面必须跟日或年；否则 market / summary / decision 里的 mar / dec 会被误当成月份
_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(?:(\d{1,2})(?:st|nd|rd|th)?\b,?\s*(\d{4})?|(\d{4})\b)",
    re.I)


def dates_in(title: str) -> set[tuple[str, str, str]]:
    """标题里出现的 (月, 日, 年)，如 'as of September 17, 2026' -> {('sep','17','2026')}。"""
    return {(m.group(1).lower(), m.group(2) or "", m.group(3) or m.group(4) or "")
            for m in _DATE_RE.finditer(title)}


def is_near_duplicate(t1: str, t2: str, simhash_max: int = 3, jaccard_min: float = 0.6) -> bool:
    k1, k2 = tokenize(t1), tokenize(t2)
    if not k1 or not k2:
        return False
    # 只差日期的模板标题（每日汇率播报、每月董事会新闻稿）不是重复：每一期都是新内容
    d1, d2 = dates_in(t1), dates_in(t2)
    if d1 and d2 and d1 != d2:
        return False
    if jaccard(k1, k2) >= jaccard_min:
        return True
    # SimHash 对过短标题不稳定，至少 5 个词才启用
    return min(len(k1), len(k2)) >= 5 and hamming(simhash(k1), simhash(k2)) <= simhash_max


# ---------- 批量去重 ----------
def dedupe_items(
    items: list[RawItem],
    known: list[tuple[str, str]],
    cfg: dict[str, Any],
) -> tuple[list[RawItem], list[tuple[RawItem, str]]]:
    """返回 (保留, [(被丢弃条目, 与之重复的条目 id)])。

    - 与库内已有条目（known）重复：丢弃
    - 批内重复：保留来源等级最高（tier 数值最小）、发布最早的一条
    """
    d = cfg.get("dedupe", {})
    smax, jmin = d.get("simhash_max_distance", 3), d.get("jaccard_threshold", 0.6)

    kept: list[RawItem] = []
    dropped: list[tuple[RawItem, str]] = []
    ordered = sorted(items, key=lambda i: (i.tier, i.published))
    for it in ordered:
        dup_of: str | None = None
        for kid, ktitle in known:
            if is_near_duplicate(it.title, ktitle, smax, jmin):
                dup_of = kid
                break
        if dup_of is None:
            for k in kept:
                if is_near_duplicate(it.title, k.title, smax, jmin):
                    dup_of = k.id
                    break
        if dup_of:
            dropped.append((it, dup_of))
        else:
            kept.append(it)
    return kept, dropped


# ---------- 事件匹配 ----------
OVERLAP_WEIGHT = 0.7


def candidate_score(a: list[str], b: list[str]) -> float:
    """候选检索分数，偏召回（最终是否同一事件由模型裁决）。

    纯 Jaccard 对「短标题 vs 长标题」的后续报道打分偏低（实测 Grab/Atome 后续报道仅 0.23，
    被漏掉），所以再取重叠系数（交集 / 较短者）加权后的值；至少共享 2 个词，避免单词误配。
    """
    sa, sb = set(a), set(b)
    shared = len(sa & sb)
    if shared < 2:
        return 0.0
    return max(jaccard(sa, sb), OVERLAP_WEIGHT * shared / min(len(sa), len(sb)))


def match_events(
    title: str,
    event_titles: dict[str, list[str]],
    candidate_threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """新条目标题 vs 各事件全部来源标题，取最大 Jaccard，按相似度降序返回 >= 候选阈值者。"""
    toks = tokenize(title)
    scored = []
    for eid, titles in event_titles.items():
        best = max((candidate_score(toks, tokenize(t)) for t in titles), default=0.0)
        if best >= candidate_threshold:
            scored.append((eid, best))
    return sorted(scored, key=lambda x: -x[1])


def event_fingerprint(country: str, title: str) -> str:
    key = country + "|" + " ".join(sorted(set(tokenize(title))))
    return hashlib.sha1(key.encode()).hexdigest()[:12]
