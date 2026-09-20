"""推送：Server 酱（微信）或邮箱，由 config.yaml 的 notify.method 决定。密钥只从环境变量读取。"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

import httpx

log = logging.getLogger("radar.notify")

SERVERCHAN_URL = "https://sctapi.ftqq.com/{key}.send"
SERVERCHAN_MAX_CHARS = 30000  # Server 酱 desp 上限约 32KB


class NotifyError(RuntimeError):
    pass


def send_serverchan(title: str, body: str, key: str | None = None) -> None:
    key = key or os.environ.get("SERVERCHAN_KEY", "")
    if not key:
        raise NotifyError("未设置 SERVERCHAN_KEY")
    if len(body) > SERVERCHAN_MAX_CHARS:
        body = body[:SERVERCHAN_MAX_CHARS] + "\n\n…（内容过长已截断，完整简报见 data/digests/）"
    try:
        r = httpx.post(SERVERCHAN_URL.format(key=key), data={"title": title[:100], "desp": body}, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        raise NotifyError(f"Server 酱请求失败: {type(exc).__name__}") from exc  # 不回显异常详情，避免带出 key
    if payload.get("code") not in (0, None):
        raise NotifyError(f"Server 酱返回错误: {payload.get('message') or payload.get('info')}")


def send_email(title: str, body: str) -> None:
    host, user, pwd, to = (os.environ.get(k, "") for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"))
    if not all((host, user, pwd, to)):
        raise NotifyError("邮件需要设置 SMTP_HOST / SMTP_USER / SMTP_PASSWORD / MAIL_TO")
    port = int(os.environ.get("SMTP_PORT", "465"))
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = title, user, to
    try:
        with smtplib.SMTP_SSL(host, port, timeout=20) as s:
            s.login(user, pwd)
            s.sendmail(user, [a.strip() for a in to.split(",")], msg.as_string())
    except Exception as exc:
        raise NotifyError(f"邮件发送失败: {type(exc).__name__}") from exc


def notify(cfg: dict[str, Any], title: str, body: str) -> str:
    """按配置推送，返回实际使用的方式（none 表示未推送）。失败抛 NotifyError，由调用方决定如何处理。"""
    method = cfg.get("notify", {}).get("method", "none")
    if method == "serverchan":
        send_serverchan(title, body)
    elif method == "email":
        send_email(title, body)
    elif method != "none":
        raise NotifyError(f"未知推送方式: {method}")
    return method
