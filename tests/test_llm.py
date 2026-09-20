import httpx
import pytest
from pydantic import BaseModel

from radar.llm import LLMClient, LLMError, LLMValidationError, parse_json


class Pong(BaseModel):
    ok: bool
    n: int


def make_client(store=None, **kw) -> LLMClient:
    return LLMClient(base_url="http://x", api_key="k", model="deepseek-chat", store=store,
                     run_id="r1", sleep=lambda s: None, **kw)


def script(client: LLMClient, outputs: list):
    """让 _call_raw 依次返回/抛出 outputs 中的元素。"""
    calls = {"n": 0}

    def fake(system, prompt):
        item = outputs[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item, 100, 20

    client._call_raw = fake
    return calls


def http_error(status: int) -> Exception:
    e = RuntimeError(f"http {status}")
    e.status_code = status
    return e


def test_parse_json_strips_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('好的，结果如下：{"a": 1} 以上') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_complete_success_logs_run(store):
    c = make_client(store)
    script(c, ['{"ok": true, "n": 3}'])
    assert c.complete("p", "s", Pong, stage="extract", prompt_version="v1") == {"ok": True, "n": 3}
    s = store.run_summary("r1")
    assert s["calls"] == 1 and s["prompt_tokens"] == 100 and s["completion_tokens"] == 20
    assert s["cost_usd"] == pytest.approx((100 * 0.28 + 20 * 0.42) / 1e6)


def test_validation_failure_retries_once_with_error_message():
    c = make_client()
    seen = []
    outs = ['{"ok": true}', '{"ok": true, "n": 1}']
    calls = script(c, outs)
    orig = c._call_raw

    def spy(system, prompt):
        seen.append(prompt)
        return orig(system, prompt)

    c._call_raw = spy
    assert c.complete("p", "s", Pong)["n"] == 1
    assert calls["n"] == 2
    assert "未通过校验" in seen[1] and "n" in seen[1]
    assert [h.ok for h in c.history] == [False, True]


def test_validation_failure_twice_raises():
    c = make_client()
    script(c, ["garbage", "still garbage"])
    with pytest.raises(LLMValidationError):
        c.complete("p", "s", Pong)


def test_retries_on_429_then_succeeds():
    c = make_client()
    sleeps = []
    c.sleep = sleeps.append
    calls = script(c, [http_error(429), http_error(503), '{"ok": true, "n": 1}'])
    assert c.complete("p", "s", Pong)["ok"] is True
    assert calls["n"] == 3 and sleeps == [1, 2]  # 指数退避


def test_gives_up_after_three_attempts():
    c = make_client()
    calls = script(c, [http_error(429)] * 5)
    with pytest.raises(LLMError):
        c.complete("p", "s", Pong)
    assert calls["n"] == 3


def test_non_retryable_error_fails_fast():
    c = make_client()
    calls = script(c, [http_error(401), '{"ok": true, "n": 1}'])
    with pytest.raises(LLMError):
        c.complete("p", "s", Pong)
    assert calls["n"] == 1


def test_timeout_is_retryable():
    class APITimeoutError(Exception):
        pass

    c = make_client()
    calls = script(c, [APITimeoutError("t"), '{"ok": true, "n": 2}'])
    assert c.complete("p", "s", Pong)["n"] == 2 and calls["n"] == 2


def test_from_env_model_override_and_profile(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "k1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setenv("LLM_ZHIPU_API_KEY", "k2")
    c = LLMClient.from_env()
    assert (c.model, c.api_key, c.provider) == ("deepseek-chat", "k1", "openai")
    c = LLMClient.from_env(model="glm-4-flash@zhipu")
    assert (c.model, c.api_key, c.base_url) == ("glm-4-flash", "k2", "https://open.bigmodel.cn/api/paas/v4")
    c = LLMClient.from_env(model="claude-sonnet-4-6")
    assert c.provider == "anthropic"


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    with pytest.raises(LLMError):
        LLMClient.from_env()


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert LLMClient.from_env(model="qwen2.5:7b").model == "qwen2.5:7b"


def test_pricing_override_from_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    c = LLMClient.from_env(model="deepseek-chat", cfg={"pricing": {"deepseek-chat": [1.0, 2.0]}})
    assert c.estimate_cost(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_unknown_model_costs_zero():
    c = make_client()
    c.model = "some-local-model"
    assert c.estimate_cost(10_000, 10_000) == 0.0
