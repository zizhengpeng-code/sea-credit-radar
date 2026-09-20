"""评估指标的正确性测试：用手工算好答案的小样本核对每个指标的定义。"""
import json

import pytest

from eval.run_eval import (aggregate, compute_metrics, load_testset, markdown_table, pct_range, run_model,
                           summary_row)
from radar.extract import ItemExtraction

from conftest import FakeLLM


def gold(i, push, cat=None, stages=(), country=None, hard=False):
    return {"id": i, "title": f"t{i}", "should_push": push, "category": cat, "impact_stage": list(stages),
            "country": country, "hard": hard}


def pred(i, relevant, cat=None, stages=(), country="ID"):
    if not relevant:
        return ItemExtraction(idx=0, relevant=False)
    return ItemExtraction(idx=0, relevant=True, country=country, category=cat, headline_cn="h",
                          why_it_matters="w", confidence=0.9, impact_stage=list(stages))


ROWS = [
    gold("a", True, "regulation", ["pricing"], "ID"),      # 全对
    gold("b", True, "product", ["limit"], "PH"),           # 推送对；类别错；环节错
    gold("c", True, "risk", ["collection"], "VN"),         # 漏报
    gold("d", False),                                      # 正确过滤
    gold("e", False, hard=True),                           # 误报（边界样本）
    gold("f", False),                                      # 正确过滤
]
PREDS = {
    "a": pred("a", True, "regulation", ["pricing"], "ID"),
    "b": pred("b", True, "market", ["acquisition"], "PH"),
    "c": pred("c", False),
    "d": pred("d", False),
    "e": pred("e", True, "funding", [], "ID"),
    "f": pred("f", False),
}


def test_push_metrics():
    m = compute_metrics(ROWS, PREDS)["push"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["miss_rate"] == pytest.approx(1 / 3)          # 应推送 3 条，漏了 1 条
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)   # 不应推送 3 条，误推 1 条


def test_category_accuracy_strict_counts_misses_as_wrong():
    c = compute_metrics(ROWS, PREDS)["category"]
    assert c["accuracy_strict"] == pytest.approx(1 / 3)          # 3 条应推送，只有 a 类别对
    assert c["accuracy_given_pushed"] == pytest.approx(1 / 2)    # 被推送的 2 条里 a 对、b 错


def test_per_class_precision_recall():
    pc = compute_metrics(ROWS, PREDS)["category"]["per_class"]
    assert (pc["regulation"]["tp"], pc["regulation"]["fp"], pc["regulation"]["fn"]) == (1, 0, 0)
    assert (pc["product"]["tp"], pc["product"]["fp"], pc["product"]["fn"]) == (0, 0, 1)   # b 应为 product 却判成 market
    assert (pc["market"]["tp"], pc["market"]["fp"]) == (0, 1)
    assert pc["funding"]["fp"] == 1            # e 是误报，其预测类别记为 funding 的 FP
    assert pc["risk"]["fn"] == 1               # c 漏报，记为 risk 的 FN
    assert pc["regulation"]["precision"] == 1.0 and pc["product"]["recall"] == 0.0


def test_stage_micro_metrics():
    s = compute_metrics(ROWS, PREDS)["impact_stage"]
    # gold 环节对：a{pricing} b{limit} c{collection}；预测：a{pricing} b{acquisition} c{}
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 2)
    assert s["precision"] == pytest.approx(1 / 2) and s["recall"] == pytest.approx(1 / 3)
    assert s["exact_match"] == pytest.approx(1 / 3)


def test_country_and_hard_accuracy():
    m = compute_metrics(ROWS, PREDS)
    assert m["country_accuracy"] == pytest.approx(1.0)      # 被推送且应推送的 a、b 国家都对
    assert m["hard_push_accuracy"] == 0.0                    # 唯一的边界样本 e 被误推


def test_failed_prediction_counts_as_miss_and_is_reported():
    preds = dict(PREDS, a=None)
    m = compute_metrics(ROWS, preds)
    assert m["failed"] == 1 and m["push"]["fn"] == 2         # a 也算漏报


def test_errors_list_contains_only_wrong_items():
    ids = {e["id"] for e in compute_metrics(ROWS, PREDS)["errors"]}
    assert ids == {"b", "c", "e"}


def test_perfect_predictions():
    rows = [gold("x", True, "risk", ["collection"], "ID"), gold("y", False)]
    preds = {"x": pred("x", True, "risk", ["collection"], "ID"), "y": pred("y", False)}
    m = compute_metrics(rows, preds)
    assert m["push"]["accuracy"] == 1.0 and m["push"]["miss_rate"] == 0.0
    assert m["category"]["accuracy_strict"] == 1.0 and m["impact_stage"]["f1"] == 1.0 and m["errors"] == []


def test_empty_stage_gold_and_pred_is_a_match():
    rows = [gold("x", True, "funding", [], "OTHER")]
    m = compute_metrics(rows, {"x": pred("x", True, "funding", [], "OTHER")})
    assert m["impact_stage"]["exact_match"] == 1.0


# ---------- 运行与聚合 ----------
def make_rows(n=3):
    return [{"id": f"i{k}", "title": f"OJK story {k} about lending", "snippet": "", "publisher": "P",
             "published": "2026-09-20", "country_hint": "ID", "tier": 3, "should_push": k == 0,
             "category": "regulation" if k == 0 else None, "impact_stage": ["pricing"] if k == 0 else [],
             "country": "ID" if k == 0 else None, "hard": False} for k in range(n)]


def test_run_model_records_cost_and_latency_with_fake_llm():
    good = {"idx": 0, "relevant": True, "country": "ID", "category": "regulation", "headline_cn": "标题",
            "why_it_matters": "原因", "impact_stage": ["pricing"], "confidence": 0.9}
    llm = FakeLLM([{"results": [good, {"idx": 1, "relevant": False}, {"idx": 2, "relevant": False}]}])
    res = run_model(None, make_rows(), {}, llm=llm)
    assert res["metrics"]["push"]["accuracy"] == 1.0 and res["prompt_version"].startswith("extract-")
    u = res["usage"]
    assert u["calls"] == 1 and u["prompt_tokens"] == 100 and u["completion_tokens"] == 20
    assert u["cost_per_item_usd"] == pytest.approx(0.001 / 3) and u["cost_per_1k_usd"] == pytest.approx(1 / 3)


def test_aggregate_mean_and_range():
    base = {"model": "m", "prompt_version": "v", "prompt_hash": "h", "testset": "t.jsonl", "n": 10,
            "stage_f1": 0.5, "latency_per_item_s": 1.0, "cost_per_1k_usd": 0.1, "failed": 0,
            "category_accuracy": 0.5, "false_alarm_rate": 0.2}
    rows = [dict(base, push_accuracy=0.8, miss_rate=0.1), dict(base, push_accuracy=0.6, miss_rate=0.3)]
    a = aggregate(rows)
    assert a["runs"] == 2 and a["push_accuracy"] == pytest.approx(0.7) and a["push_accuracy_min"] == 0.6
    assert pct_range(a, "push_accuracy") == "70.0% (60-80)"
    assert pct_range(aggregate([rows[0]]), "push_accuracy") == "80.0%"
    assert "70.0% (60-80)" in markdown_table([a])


def test_shipped_testsets_are_valid_and_disjoint():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "eval"
    dev, hold = load_testset(root / "testset.jsonl"), load_testset(root / "holdout.jsonl")
    assert len(dev) >= 40
    for rows in (dev, hold):
        for r in rows:
            assert r["should_push"] in (True, False)
            if r["should_push"]:
                assert r["category"] in ("regulation", "product", "funding", "market", "risk")
                assert set(r["impact_stage"]) <= {"acquisition", "underwriting", "limit", "pricing", "collection"}
                assert r["country"] in ("ID", "PH", "SG", "VN", "OTHER")
    assert not ({r["title"] for r in dev} & {r["title"] for r in hold})   # 留出集与开发集不重叠
    assert json.dumps(dev[0], ensure_ascii=False)                          # 可序列化


def test_summary_row_shape():
    rec = {"timestamp": "t", "run": 1, "testset": {"file": "x.jsonl", "sha": "abc"}, "model": "m",
           "prompt_version": "v", "prompt_hash": "h", "metrics": compute_metrics(ROWS, PREDS),
           "usage": {"latency_per_item_s": 0.5, "cost_per_1k_usd": 0.1}}
    r = summary_row(rec)
    assert r["miss_rate"] == pytest.approx(1 / 3) and r["testset"] == "x.jsonl" and r["failed"] == 0
