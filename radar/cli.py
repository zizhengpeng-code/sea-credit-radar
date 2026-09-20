"""命令行入口：python -m radar.cli <run|backfill|fetch|stats> ..."""
from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from .config import load_config
from .pipeline import ingest
from .store import Store

console = Console()


def cmd_fetch(args, cfg) -> int:
    store = Store(cfg["db_path"])
    res = ingest(cfg, store, window_days=None)
    rep = res.pop("feed_report")
    t = Table(title="抓取结果")
    t.add_column("指标")
    t.add_column("数量", justify="right")
    for k, label in [("fetched", "抓取条目（窗口内）"), ("new_url", "新 URL"),
                     ("near_duplicates", "标题近似重复"), ("to_process", "待处理")]:
        t.add_row(label, str(res[k]))
    console.print(t)
    for name, n in rep["per_feed"].items():
        console.print(f"  {name}: {n}", style="red" if name in rep["errors"] else None)
    for name, err in rep["errors"].items():
        console.print(f"[red]失败[/red] {name}: {err}")
    return 0


def cmd_stats(args, cfg) -> int:
    store = Store(cfg["db_path"])
    console.print(store.count_items())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radar", description="SEA Credit Radar")
    p.add_argument("--config", default=None, help="config.yaml 路径")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="仅抓取并入库（阶段一）").set_defaults(fn=cmd_fetch)
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
    return args.fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
