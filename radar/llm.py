"""统一大模型 client。业务代码只通过 LLMClient.complete() 调用模型，不直接接触任何厂商 SDK。

- 默认走 OpenAI 兼容接口（DeepSeek / 智谱 / 通义 / Ollama 只改三个环境变量）
- Claude 在 client 内部走 anthropic SDK 的适配分支
- 只用纯文本 JSON 输出：不依赖 function calling / tool use / response_format，换模型即可用
- 超时、429、5xx 指数退避重试（最多 3 次尝试）；JSON 解析或 pydantic 校验失败带错误信息重试一次
- 每次调用记录 token、估算成本、延迟，写入 runs 表
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

log = logging.getLogger("radar.llm")

# 单价：美元 / 百万 token，(输入, 输出)。仅用于成本估算，价格会变，可在 config.yaml 的 pricing 覆盖。
PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),
    "glm-4-flash": (0.0, 0.0),
    "glm-4-plus": (0.70, 0.70),
    "qwen-plus": (0.40, 1.20),
    "qwen-turbo": (0.05, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class LLMError(RuntimeError):
    """调用失败（网络/限流重试耗尽）。"""


class LLMValidationError(LLMError):
    """模型输出经一次重试仍无法通过 JSON 解析或 schema 校验。"""


@dataclass
class CallRecord:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    ok: bool
    note: str = ""


@dataclass
class LLMClient:
    base_url: str
    api_key: str
    model: str
    provider: str = "openai"  # openai（兼容接口）| anthropic
    store: Any = None  # radar.store.Store，可为 None
    run_id: str = "adhoc"
    timeout: float = 90.0
    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(PRICING))
    sleep: Callable[[float], None] = time.sleep
    history: list[CallRecord] = field(default_factory=list)
    _sdk: Any = None

    # ---------- 构造 ----------
    @classmethod
    def from_env(cls, model: str | None = None, store: Any = None, run_id: str = "adhoc",
                 cfg: dict[str, Any] | None = None) -> "LLMClient":
        """从环境变量构造。`model` 参数（对应 --model）临时覆盖 LLM_MODEL。

        只写模型名时，base_url / key 仍取 LLM_BASE_URL / LLM_API_KEY。
        跨供应商对比用 `模型名@档案名`（如 `glm-4-flash@zhipu`），此时读取
        LLM_ZHIPU_BASE_URL / LLM_ZHIPU_API_KEY（可选 LLM_ZHIPU_PROVIDER），互不干扰。
        """
        model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        base_url, api_key = os.environ.get("LLM_BASE_URL", ""), os.environ.get("LLM_API_KEY", "")
        provider = os.environ.get("LLM_PROVIDER", "")
        # 支持 LLM_PROFILE_<NAME>_BASE_URL / _API_KEY：--model name@profile 时切换整套凭据（便于多供应商对比）
        if "@" in model:
            model, profile = model.split("@", 1)
            p = profile.upper()
            base_url = os.environ.get(f"LLM_{p}_BASE_URL", base_url)
            api_key = os.environ.get(f"LLM_{p}_API_KEY", api_key)
            provider = os.environ.get(f"LLM_{p}_PROVIDER", provider)
        if not provider:
            provider = "anthropic" if model.startswith("claude") else "openai"
        if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
            raise LLMError("未设置 LLM_API_KEY（见 .env.example）")
        pricing = dict(PRICING)
        for k, v in (cfg or {}).get("pricing", {}).items():
            pricing[k] = (float(v[0]), float(v[1]))
        return cls(base_url=base_url, api_key=api_key or "ollama", model=model,
                   provider=provider, store=store, run_id=run_id, pricing=pricing)

    def _client(self) -> Any:
        if self._sdk is None:
            if self.provider == "anthropic":
                import anthropic

                kw: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 0}
                if self.base_url and "anthropic.com" not in self.base_url:
                    kw["base_url"] = self.base_url
                self._sdk = anthropic.Anthropic(**kw)
            else:
                from openai import OpenAI

                self._sdk = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                   timeout=self.timeout, max_retries=0)  # 重试由本类统一处理
        return self._sdk

    # ---------- 底层调用 ----------
    def _call_raw(self, system: str, prompt: str) -> tuple[str, int, int]:
        c = self._client()
        if self.provider == "anthropic":
            r = c.messages.create(model=self.model, max_tokens=4096, temperature=0, system=system,
                                  messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens
        r = c.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        u = getattr(r, "usage", None)
        return (r.choices[0].message.content or "",
                getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status == 429 or status >= 500
        name = type(exc).__name__.lower()
        return any(k in name for k in ("timeout", "connection", "ratelimit"))

    def _call_with_backoff(self, system: str, prompt: str, attempts: int = 3) -> tuple[str, int, int]:
        for i in range(attempts):
            try:
                return self._call_raw(system, prompt)
            except Exception as exc:
                if not self._is_retryable(exc) or i == attempts - 1:
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                delay = 2 ** i  # 1s, 2s
                log.warning("LLM 调用失败（%s），%ss 后重试 %d/%d", type(exc).__name__, delay, i + 1, attempts - 1)
                self.sleep(delay)
        raise LLMError("unreachable")

    # ---------- 成本 ----------
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = self.pricing.get(self.model, (0.0, 0.0))
        return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000

    def _record(self, stage: str, pt: int, ct: int, latency: float, ok: bool,
                prompt_version: str, note: str = "") -> None:
        cost = self.estimate_cost(pt, ct)
        self.history.append(CallRecord(stage, self.model, pt, ct, cost, latency, ok, note))
        if self.store is not None:
            self.store.log_run(run_id=self.run_id, stage=stage, model=self.model,
                               prompt_version=prompt_version, prompt_tokens=pt,
                               completion_tokens=ct, cost_usd=cost, latency_s=latency,
                               ok=ok, note=note[:300])

    # ---------- 统一入口 ----------
    def complete(self, prompt: str, system: str, schema: type[BaseModel], *,
                 stage: str = "llm", prompt_version: str = "") -> dict[str, Any]:
        """调用模型并返回经 schema 校验的 dict。校验失败带错误信息重试一次。"""
        user_prompt = prompt
        last_err = ""
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                text, pt, ct = self._call_with_backoff(system, user_prompt)
            except LLMError as exc:
                self._record(stage, 0, 0, time.perf_counter() - t0, False, prompt_version, str(exc))
                raise
            latency = time.perf_counter() - t0
            try:
                data = schema.model_validate(parse_json(text)).model_dump(mode="json")
            except (ValueError, ValidationError) as exc:
                last_err = _short_error(exc)
                self._record(stage, pt, ct, latency, False, prompt_version, f"invalid: {last_err}")
                user_prompt = (
                    f"{prompt}\n\n【上一次输出未通过校验】\n错误：{last_err}\n"
                    f"上一次输出（节选）：{text[:1200]}\n"
                    "请修正后重新输出完整 JSON，不要任何前言、解释或 Markdown 代码块。"
                )
                continue
            self._record(stage, pt, ct, latency, True, prompt_version)
            return data
        raise LLMValidationError(f"重试后仍无法通过校验：{last_err}")


# ---------- JSON 解析 ----------
def parse_json(text: str) -> Any:
    """容错解析：去掉可能出现的 ``` 围栏与前后说明，取第一个 { 到最后一个 }。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出中没有 JSON 对象")
    return json.loads(t[start: end + 1])


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]]
        return "; ".join(parts)
    return str(exc)[:300]
