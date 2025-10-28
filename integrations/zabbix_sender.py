"""Helper utilities for sending metrics to Zabbix via ``zabbix_sender``."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_TIMEOUT_SECONDS = 10.0


class ZabbixSenderError(RuntimeError):
    """Raised when ``zabbix_sender`` fails to send an event."""


@dataclass
class _ZabbixConfigView:
    """A lightweight view over the configuration used by :func:`send_event`."""

    zabbix_sender_path: Optional[str]
    zabbix_server: Optional[str]
    zabbix_host: Optional[str]
    zabbix_key: Optional[str]


def _normalize_config(config: Any) -> _ZabbixConfigView:
    return _ZabbixConfigView(
        zabbix_sender_path=getattr(config, "zabbix_sender_path", None),
        zabbix_server=(getattr(config, "zabbix_server", None) or None),
        zabbix_host=(getattr(config, "zabbix_host", None) or None),
        zabbix_key=(getattr(config, "zabbix_key", None) or None),
    )


def _resolve_sender_path(cfg: _ZabbixConfigView) -> str:
    if cfg.zabbix_sender_path:
        candidate = cfg.zabbix_sender_path
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        resolved_candidate = shutil.which(candidate)
        if resolved_candidate:
            return resolved_candidate
    resolved = shutil.which("zabbix_sender")
    if resolved:
        return resolved
    raise ZabbixSenderError("Бинарник zabbix_sender не найден в системе")


def send_event(summary: str, severity: int, host: str, count: int, config: Any, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """Send an incident summary to Zabbix using ``zabbix_sender``.

    Parameters
    ----------
    summary:
        Short text produced by the LLM describing the incident.
    severity:
        Syslog severity level.
    host:
        Host from which the event originated.
    count:
        Number of repeated messages aggregated into the incident.
    config:
        Object exposing Zabbix-specific attributes from :class:`agent_daemon.Config`.
    timeout:
        Timeout passed to :func:`subprocess.run`.
    """

    cfg = _normalize_config(config)
    if not cfg.zabbix_server:
        raise ZabbixSenderError("Не указан адрес сервера Zabbix")
    if not cfg.zabbix_key:
        raise ZabbixSenderError("Не указан ключ метрики Zabbix")

    sender_path = _resolve_sender_path(cfg)
    metric_value = json.dumps(
        {
            "summary": summary,
            "severity": severity,
            "host": host,
            "count": count,
        },
        ensure_ascii=False,
    )

    zabbix_host = cfg.zabbix_host or host
    args = [
        sender_path,
        "-z",
        cfg.zabbix_server,
        "-s",
        zabbix_host,
        "-k",
        cfg.zabbix_key,
        "-o",
        metric_value,
    ]

    logging.debug("Отправка события в Zabbix: %s", metric_value)

    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ZabbixSenderError("zabbix_sender завершился по таймауту") from exc
    except FileNotFoundError as exc:
        raise ZabbixSenderError("zabbix_sender не найден") from exc
    except subprocess.CalledProcessError as exc:
        message = f"zabbix_sender завершился с кодом {exc.returncode}: {exc.stderr or exc.stdout}"
        raise ZabbixSenderError(message.strip()) from exc

    if completed.stdout:
        logging.debug("Ответ zabbix_sender: %s", completed.stdout.strip())
    if completed.stderr:
        logging.warning("zabbix_sender stderr: %s", completed.stderr.strip())

