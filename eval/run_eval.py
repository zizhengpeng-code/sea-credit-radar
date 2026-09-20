"""评估脚本：在人工标注测试集上量化抽取质量，并支持多模型、多 prompt 版本对比。

用法：
  python eval/run_eval.py                                   # 默认模型 + 默认 prompt
  python eval/run_eval.py --model deepseek-chat glm-4-flash@zhipu   # 多模型对比（同一套测试集）
  python eval/run_eval.py --prompt-version extract-v1 extract-v2   # prompt A/B
  python eval/run_eval.py --testset eval/holdout.jsonl --repeat 3  # 留出集，重复 3 次取均值
  python eval/run_eval.py --history                          # 汇总历次结果

只评估 extract 阶段（相关性 / 类别 / 影响环节 / 国家），不含事件归并与 enrich。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from radar.config import load_config  # noqa: E402
from radar.extract import ItemExtraction, extract_all  # noqa: E402
from radar.llm import LLMClient, LLMError  # noqa: E402
from radar.models import CATEGORIES, RawItem  # noqa: E402
from radar.prompts import EXTRACT_PROMPTS, get_extract_system  # noqa: E402

console = Console()
TESTSET = ROOT / "eval" / "testset.jsonl"
RESULTS = ROOT / "eval" / "results"


# ---------- 数据 ----------
def load_testset(path: Path = TESTSET) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("测试集 id 重复")
    return rows


def to_raw_item(row: dict[str, Any]) -> RawItem:
    pub = dt.datetime.fromisoformat(row["published"]).replace(tzinfo=dt.timezone.utc)
    return RawItem(id=row["id"], url=f"eval://{row['id']}", title=row["title"], publisher=row.get("publisher", ""),
                   published=pub, fetched_at=pub, feed_name="eval", country_hint=row.get("country_hint", "OTHER"),
                   tier=row.get("tier", 3), snippet=row.get("snippet", ""))


def file_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


# ---------- 指标 ----------
def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(rows: list[dict[str, Any]], preds: dict[str, ItemExtraction | None]) -> dict[str, Any]:
    """preds: 测试集 id -> 抽取结果；None 表示模型调用失败 / 未返回（按「未推送」计，并单独统计）。"""
    n = len(rows)
    tp = fp = fn = tn = failed = 0
    cat_correct = cat_conditional_total = 0
    country_correct = 0
    stage_tp = stage_fp = stage_fn = stage_exact = 0
    per_cat = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    errors: list[dict[str, Any]] = []
    n_pos = sum(r["should_push"] for r in rows)

    for r in rows:
        p = preds.get(r["id"])
        if p is None:
            failed += 1
        pushed = bool(p and p.relevant)
        gold_push = r["should_push"]
        if gold_push and pushed:
            tp += 1
        elif gold_push and not pushed:
            fn += 1
        elif not gold_push and pushed:
            fp += 1
        else:
            tn += 1

        pred_cat = p.category if pushed else None
        gold_cat = r["category"] if gold_push else None
        # 各类别混淆：只在「应推送或被推送」的条目上统计
        if gold_cat and pred_cat == gold_cat:
            per_cat[gold_cat]["tp"] += 1
        else:
            if gold_cat:
                per_cat[gold_cat]["fn"] += 1
            if pred_cat:
                per_cat[pred_cat]["fp"] += 1

        if gold_push:
            gold_st = set(r["impact_stage"])
            pred_st = set(p.impact_stage) if pushed else set()
            stage_tp += len(gold_st & pred_st)
            stage_fp += len(pred_st - gold_st)
            stage_fn += len(gold_st - pred_st)
            stage_exact += int(gold_st == pred_st and pushed)
            if pushed:
                cat_conditional_total += 1
                cat_correct += int(pred_cat == gold_cat)
                country_correct += int(p.country == r["country"])

        wrong = (gold_push != pushed) or (gold_push and pushed and (
            pred_cat != gold_cat or set(p.impact_stage) != set(r["impact_stage"])))
        if wrong:
            errors.append({
                "id": r["id"], "title": r["title"], "gold": {"push": gold_push, "category": gold_cat,
                                                             "impact_stage": r["impact_stage"]},
                "pred": None if p is None else {"push": pushed, "category": pred_cat,
                                                "impact_stage": p.impact_stage if pushed else []},
                "hard": r.get("hard", False)})

    hard = [r for r in rows if r.get("hard")]
    hard_ok = sum(1 for r in hard if bool(preds.get(r["id"]) and preds[r["id"]].relevant) == r["should_push"])
    return {
        "n": n, "n_push": n_pos, "n_not_push": n - n_pos, "failed": failed,
        "push": {"accuracy": (tp + tn) / n, "precision": tp / (tp + fp) if tp + fp else 0.0,
                 "recall": tp / (tp + fn) if tp + fn else 0.0,
                 "miss_rate": fn / n_pos if n_pos else 0.0,          # 漏报率：应推送却被过滤
                 "false_alarm_rate": fp / (n - n_pos) if n - n_pos else 0.0,
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "category": {"accuracy_strict": cat_correct / n_pos if n_pos else 0.0,   # 分母=应推送总数，漏报算错
                     "accuracy_given_pushed": cat_correct / cat_conditional_total if cat_conditional_total else 0.0,
                     "per_class": {c: _prf(**v) for c, v in per_cat.items()}},
        "impact_stage": {**_prf(stage_tp, stage_fp, stage_fn),
                         "exact_match": stage_exact / n_pos if n_pos else 0.0},
        "country_accuracy": country_correct / cat_conditional_total if cat_conditional_total else 0.0,
        "hard_push_accuracy": hard_ok / len(hard) if hard else None,
        "errors": errors,
    }


# ---------- 运行 ----------
def prompt_hash(version: str) -> str:
    return hashlib.sha1(get_extract_system(version)[1].encode()).hexdigest()[:8]


def run_model(model: str | None, rows: list[dict[str, Any]], cfg: dict[str, Any], batch_size: int = 8,
              llm: LLMClient | None = None, version: str | None = None) -> dict[str, Any]:
    version = get_extract_system(version)[0]
    llm = llm or LLMClient.from_env(model=model, cfg=cfg, run_id="eval")
    items = [to_raw_item(r) for r in rows]
    t_start = time.perf_counter()
    results = extract_all(llm, items, batch_size, version=version)  # 与生产同一条路径：逐条校验 + 失败拆批
    wall = time.perf_counter() - t_start
    preds = {it.id: x for it, x, _ in results}

    n = len(rows)
    pt = sum(c.prompt_tokens for c in llm.history)
    ct = sum(c.completion_tokens for c in llm.history)
    cost = sum(c.cost_usd for c in llm.history)
    calls = len(llm.history)
    return {
        "model": llm.model, "provider": llm.provider, "base_url": llm.base_url, "prompt_version": version,
        "metrics": compute_metrics(rows, preds),
        "usage": {"calls": calls, "calls_ok": sum(c.ok for c in llm.history),
                  "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
                  "cost_per_item_usd": cost / n, "cost_per_1k_usd": cost / n * 1000,
                  "latency_per_item_s": wall / n, "latency_per_call_s": wall / calls if calls else 0.0,
                  "wall_s": wall, "call_failures": [f"{c.stage}: {c.note}" for c in llm.history if not c.ok]},
    }


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def save_result(res: dict[str, Any], ts: str, testset: Path, note: str = "", run_idx: int = 1) -> dict[str, Any]:
    """写单次结果 JSON 与 history.jsonl，返回带元信息的完整记录。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    ver = res["prompt_version"]
    rec = {**res, "timestamp": ts, "run": run_idx, "prompt_hash": prompt_hash(ver),
           "testset": {"file": testset.name, "sha": file_sha(testset), "n": res["metrics"]["n"]}, "note": note}
    path = RESULTS / f"{ts}_{testset.stem}_{safe(res['model'])}_{ver}_run{run_idx}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(RESULTS / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_row(rec), ensure_ascii=False) + "\n")
    rec["_path"] = path
    return rec


NUMERIC = ("push_accuracy", "category_accuracy", "miss_rate", "false_alarm_rate", "stage_f1",
           "latency_per_item_s", "cost_per_1k_usd", "failed")


def summary_row(rec: dict[str, Any]) -> dict[str, Any]:
    m, u = rec["metrics"], rec["usage"]
    return {"timestamp": rec["timestamp"], "run": rec.get("run", 1), "testset": rec["testset"].get("file", ""),
            "model": rec["model"], "prompt_version": rec["prompt_version"], "prompt_hash": rec["prompt_hash"],
            "testset_sha": rec["testset"]["sha"], "n": m["n"],
            "push_accuracy": m["push"]["accuracy"], "category_accuracy": m["category"]["accuracy_strict"],
            "miss_rate": m["push"]["miss_rate"], "false_alarm_rate": m["push"]["false_alarm_rate"],
            "stage_f1": m["impact_stage"]["f1"], "latency_per_item_s": u["latency_per_item_s"],
            "cost_per_1k_usd": u["cost_per_1k_usd"], "failed": m["failed"], "note": rec.get("note", "")}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """同一（模型, prompt, 测试集）的多次运行取均值，并保留最小/最大值以反映采样波动。"""
    out = dict(rows[0])
    out["runs"] = len(rows)
    for k in NUMERIC:
        vals = [r[k] for r in rows]
        out[k] = sum(vals) / len(vals)
        out[k + "_min"], out[k + "_max"] = min(vals), max(vals)
    return out


# ---------- 展示 ----------
def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def pct_range(r: dict[str, Any], key: str) -> str:
    """多次运行时显示「均值 (最小-最大)」，单次只显示值。"""
    if r.get("runs", 1) > 1 and r.get(key + "_min") != r.get(key + "_max"):
        return f"{r[key] * 100:.1f}% ({r[key + '_min'] * 100:.0f}-{r[key + '_max'] * 100:.0f})"
    return pct(r[key])


COLUMNS = ["模型", "测试集", "提示词", "推送准确率", "分类准确率", "漏报率", "误报率", "环节 F1",
           "延迟 s/条", "每千条成本 $", "运行", "失败"]


def _cells(r: dict[str, Any]) -> list[str]:
    runs = r.get("runs", 1)
    return [r["model"], r.get("testset", "").replace(".jsonl", ""), f'{r["prompt_version"]}·{r["prompt_hash"]}',
            pct_range(r, "push_accuracy"), pct_range(r, "category_accuracy"), pct_range(r, "miss_rate"),
            pct_range(r, "false_alarm_rate"), pct_range(r, "stage_f1"), f'{r["latency_per_item_s"]:.2f}',
            f'{r["cost_per_1k_usd"]:.3f}', str(runs), f'{r["failed"]:.1f}' if runs > 1 else str(int(r["failed"]))]


def compare_table(rows: list[dict[str, Any]], title: str = "对比") -> Table:
    t = Table(title=title)
    for i, col in enumerate(COLUMNS):
        t.add_column(col, justify="left" if i < 3 else "right")
    for r in rows:
        t.add_row(*_cells(r))
    return t


def markdown_table(rows: list[dict[str, Any]]) -> str:
    head = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
    return head + "".join("| " + " | ".join(_cells(r)) + " |\n" for r in rows)


def print_detail(res: dict[str, Any], show_errors: int) -> None:
    m, u = res["metrics"], res["usage"]
    console.rule(f"[bold]{res['model']} · {res['prompt_version']}[/bold]  n={m['n']}"
                 f"（应推送 {m['n_push']} / 不应推送 {m['n_not_push']}）")
    p = m["push"]
    console.print(f"推送判断：准确率 {pct(p['accuracy'])} | 精确率 {pct(p['precision'])} | 召回率 {pct(p['recall'])} | "
                  f"[bold]漏报率 {pct(p['miss_rate'])}[/bold]（{p['fn']}/{m['n_push']}）| 误报率 {pct(p['false_alarm_rate'])}"
                  f"（{p['fp']}/{m['n_not_push']}）| 边界样本判对率 {pct(m['hard_push_accuracy'])}")
    c = m["category"]
    console.print(f"类别：严格准确率 {pct(c['accuracy_strict'])}（漏报算错）| 已推送条目中的准确率 "
                  f"{pct(c['accuracy_given_pushed'])} | 国家准确率 {pct(m['country_accuracy'])}")
    t = Table(title="各类别")
    for col in ("类别", "精确率", "召回率", "F1", "TP/FP/FN"):
        t.add_column(col, justify="right" if col != "类别" else "left")
    for k, v in c["per_class"].items():
        t.add_row(k, pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), f'{v["tp"]}/{v["fp"]}/{v["fn"]}')
    console.print(t)
    s = m["impact_stage"]
    console.print(f"影响环节（微平均）：P {pct(s['precision'])} R {pct(s['recall'])} F1 {pct(s['f1'])} | "
                  f"整体完全匹配 {pct(s['exact_match'])}")
    for f in u.get("call_failures", []):
        console.print(f"  [yellow]校验失败[/yellow] {f[:160]}")
    console.print(f"调用 {u['calls']} 次（成功 {u['calls_ok']}）| tokens {u['prompt_tokens']}+{u['completion_tokens']} | "
                  f"成本 ${u['cost_usd']:.4f}（每千条 ${u['cost_per_1k_usd']:.3f}）| "
                  f"延迟 {u['latency_per_item_s']:.2f}s/条（{u['latency_per_call_s']:.1f}s/次）| 失败条目 {m['failed']}")
    if show_errors and m["errors"]:
        console.print(f"[bold]错误样本（前 {show_errors} 条）[/bold]")
        for e in m["errors"][:show_errors]:
            console.print(f"  {'[边界]' if e['hard'] else '      '} {e['id']} {e['title'][:70]}\n"
                          f"        标注 {e['gold']}\n        预测 {e['pred']}")


def load_history() -> list[dict[str, Any]]:
    path = RESULTS / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    console.file = sys.stdout
    ap = argparse.ArgumentParser(description="SEA Credit Radar 评估")
    ap.add_argument("--model", nargs="+", default=[None], help="一个或多个模型；跨供应商用 模型名@档案名")
    ap.add_argument("--prompt-version", nargs="+", default=[None],
                    help=f"一个或多个 prompt 版本，可选 {sorted(EXTRACT_PROMPTS)}；默认当前版本")
    ap.add_argument("--testset", default=str(TESTSET))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1, help="重复运行次数（模型输出有采样波动，取均值更可靠）")
    ap.add_argument("--errors", type=int, default=0, help="打印前 N 条错误样本（仅最后一次运行）")
    ap.add_argument("--note", default="", help="写入结果的备注（例如本次 prompt 改了什么）")
    ap.add_argument("--history", action="store_true", help="只汇总历次结果，不运行")
    args = ap.parse_args(argv)

    if args.history:
        hist = load_history()
        if not hist:
            console.print("还没有历史结果")
            return 0
        console.print(compare_table(hist, "历次评估（每次运行一行）"))
        return 0

    cfg = load_config()
    testset = Path(args.testset)
    rows = load_testset(testset)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    summaries = []
    for model in args.model:
        for version in args.prompt_version:
            runs = []
            for k in range(1, args.repeat + 1):
                try:
                    res = run_model(model, rows, cfg, args.batch_size, version=version)
                except (LLMError, KeyError) as exc:
                    console.print(f"[red]{model or '默认模型'} / {version or '默认 prompt'} 无法运行：{exc}[/red]")
                    break
                rec = save_result(res, ts, testset, args.note, k)
                runs.append(summary_row(rec))
                if k == args.repeat:
                    print_detail(res, args.errors)
                console.print(f"[dim]{res['model']} {res['prompt_version']} 第 {k}/{args.repeat} 次："
                              f"推送准确率 {pct(runs[-1]['push_accuracy'])} 漏报率 {pct(runs[-1]['miss_rate'])} "
                              f"失败 {runs[-1]['failed']} → {rec['_path'].relative_to(ROOT)}[/dim]")
            if runs:
                summaries.append(aggregate(runs))
    if summaries:
        console.print(compare_table(summaries))
        out = RESULTS / f"{ts}_{testset.stem}_compare.md"
        out.write_text(f"# 评估对比 {ts}\n\n测试集 {testset.name}（n={len(rows)}），每组运行 {args.repeat} 次取均值"
                       f"（括号内为最小-最大）\n\n" + markdown_table(summaries), encoding="utf-8")
        console.print(f"对比表已写入 {out.relative_to(ROOT)}")
    return 0 if summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
