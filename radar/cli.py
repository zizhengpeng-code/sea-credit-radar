"""命令行入口：python -m radar.cli <run|backfill|digest|fetch|extract|stats> ..."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import load_config, today
from .digest import build_digest, render_markdown
from .llm import LLMClient, LLMError
from .pipeline import backfill, ingest, process, run_pipeline
from .store import Store

console = Console()


def _run_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now():%Y%m%d-%H%M%S}"


def _cost_line(store: Store, llm: LLMClient) -> str:
    s = store.run_summary(llm.run_id)
    return (f"模型 {llm.model} | 调用 {s['calls']} 次 | tokens {s['prompt_tokens']}+{s['completion_tokens']}"
            f" | 估算成本 ${s['cost_usd']:.4f}")


def _print_ingest(res: dict) -> None:
    rep = res.get("feed_report", {})
    t = Table(title="抓取结果")
    t.add_column("指标")
    t.add_column("数量", justify="right")
    for k, label in [("fetched", "抓取条目（窗口内）"), ("new_url", "新 URL"),
                     ("near_duplicates", "标题近似重复"), ("merged_into_events", "并入既有事件来源"),
                     ("to_process", "待处理")]:
        if k in res:
            t.add_row(label, str(res[k]))
    console.print(t)
    for name, err in rep.get("errors", {}).items():
        console.print(f"[red]抓取失败[/red] {name}: {err}")


def cmd_fetch(args, cfg) -> int:
    _print_ingest(ingest(cfg, Store(cfg["db_path"])))
    return 0


def cmd_extract(args, cfg) -> int:
    store = Store(cfg["db_path"])
    llm = LLMClient.from_env(model=args.model, store=store, run_id=_run_id("extract"), cfg=cfg)
    console.print(process(cfg, store, llm, today(cfg), limit=args.limit))
    console.print(_cost_line(store, llm))
    return 0


def cmd_run(args, cfg) -> int:
    store = Store(cfg["db_path"])
    llm = LLMClient.from_env(model=args.model, store=store, run_id=_run_id("run"), cfg=cfg)
    res = run_pipeline(cfg, store, llm, today(cfg), dry_run=args.dry_run, limit=args.limit)
    _print_ingest(res.ingest_stats)
    console.print(dict(res.process_stats))
    console.print(Markdown(res.markdown))
    console.print(f"简报已保存：{res.digest_path}")
    console.print(_cost_line(store, llm))
    if args.dry_run:
        console.print("[yellow]--dry-run：未推送[/yellow]")
    elif res.notify_error:
        console.print(f"[red]推送失败[/red]：{res.notify_error}（简报已落盘）")
        return 3
    elif res.notified == "none":
        console.print("推送方式为 none（config.yaml 的 notify.method），未推送")
    else:
        console.print(f"[green]已推送[/green]（{res.notified}）")
    return 0


def cmd_backfill(args, cfg) -> int:
    store = Store(cfg["db_path"])
    llm = LLMClient.from_env(model=args.model, store=store, run_id=_run_id("backfill"), cfg=cfg)
    res = backfill(cfg, store, llm, args.days)
    _print_ingest(res.pop("ingest"))
    console.print(res)
    console.print(_cost_line(store, llm))
    return 0


def cmd_digest(args, cfg) -> int:
    """从库中重新渲染某天的简报（不联网、不调用模型）。"""
    store = Store(cfg["db_path"])
    day = dt.date.fromisoformat(args.date) if args.date else today(cfg)
    md = render_markdown(build_digest(store, day))
    console.print(Markdown(md))
    if args.export_svg:  # 导出终端渲染效果，用作 README 示例截图
        rec = Console(record=True, width=100, file=open(os.devnull, "w", encoding="utf-8"))
        rec.print(Markdown(md))
        Path(args.export_svg).parent.mkdir(parents=True, exist_ok=True)
        rec.save_svg(args.export_svg, title=f"SEA Credit Radar {day.isoformat()}")
        console.print(f"已导出 {args.export_svg}")
    return 0


def cmd_stats(args, cfg) -> int:
    console.print(Store(cfg["db_path"]).count_items())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radar", description="SEA Credit Radar")
    p.add_argument("--config", default=None, help="config.yaml 路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑全流程：抓取 -> 抽取 -> 简报 -> 推送")
    r.add_argument("--dry-run", action="store_true", help="不推送，只在终端打印")
    r.add_argument("--model", default=None, help="临时覆盖 LLM_MODEL")
    r.add_argument("--limit", type=int, default=None, help="本次最多处理的条目数")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("backfill", help="回填历史：按发布日期逐日处理，不推送")
    b.add_argument("--days", type=int, required=True)
    b.add_argument("--model", default=None)
    b.set_defaults(fn=cmd_backfill)

    d = sub.add_parser("digest", help="从库中重新渲染某天简报（默认今天）")
    d.add_argument("--date", default=None, help="YYYY-MM-DD")
    d.add_argument("--export-svg", default=None, help="把终端渲染效果导出为 SVG")
    d.set_defaults(fn=cmd_digest)

    sub.add_parser("fetch", help="仅抓取并入库").set_defaults(fn=cmd_fetch)
    ex = sub.add_parser("extract", help="仅对 status=new 的条目做抽取")
    ex.add_argument("--limit", type=int, default=None)
    ex.add_argument("--model", default=None)
    ex.set_defaults(fn=cmd_extract)
    sub.add_parser("stats", help="库内条目状态统计").set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows 默认 GBK，中文会乱码
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    console.file = sys.stdout
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg)
    except LLMError as exc:
        console.print(f"[red]模型错误[/red]：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
