"""配置加载：config.yaml + .env。"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    method = os.environ.get("NOTIFY_METHOD")  # CI 里用环境变量选择推送方式，不必改 config.yaml
    if method:
        cfg.setdefault("notify", {})["method"] = method
    db = Path(cfg.get("db_path", "data/radar.db"))
    cfg["db_path"] = str(db if db.is_absolute() else ROOT / db)
    return cfg


def local_tz(cfg: dict[str, Any]) -> dt.timezone:
    return dt.timezone(dt.timedelta(hours=cfg.get("timezone_offset_hours", 8)))


def today(cfg: dict[str, Any]) -> dt.date:
    return dt.datetime.now(local_tz(cfg)).date()
