# -*- coding: utf-8 -*-
"""
plugins/xorg.py
Плагин для логов Xorg: выбирает строки с тегом xorg: и парсит только ошибки (EE).

Контракт:
- can_handle(log_line: str) -> bool
- parse(log_line: str) -> Optional[Dict]

Возвращаемый словарь:
{ "original_line", "severity", "count", "message", "host" }
"""
from typing import Optional, Dict
import re

PRIORITY = 100  # выше, чем у syslog_default

RE_REPEAT = re.compile(
    r"<(\d+)>\s*(?:\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})?.*?message repeated\s+(\d+)\s+times:\s*\[\s*(.*)\s*\]",
    re.IGNORECASE,
)
RE_STANDARD = re.compile(r"<(\d+)>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)")


def _split_tag_and_message(text: str):
    parts = text.split(":", 1)
    tag = parts[0].strip() if parts else ""
    msg = parts[1].strip() if len(parts) > 1 else text
    return tag, msg


def _is_xorg_error(tag: str, message: str) -> bool:
    if not tag.lower().startswith("xorg"):
        return False
    # Ошибки Xorg имеют маркер "(EE)" в сообщении
    return "(EE)" in message


def can_handle(line: str) -> bool:
    # Повторяющиеся сообщения
    m = RE_REPEAT.search(line)
    if m:
        inner = m.group(3).strip()
        tag, msg = _split_tag_and_message(inner)
        return _is_xorg_error(tag, msg)

    # Обычная строка
    m = RE_STANDARD.search(line)
    if not m:
        return False
    full_message = m.group(4).strip()
    tag, msg = _split_tag_and_message(full_message)
    return _is_xorg_error(tag, msg)


def parse(line: str) -> Optional[Dict]:
    m = RE_REPEAT.search(line)
    if m:
        severity = int(m.group(1)) & 7
        count = int(m.group(2))
        inner = m.group(3).strip()
        tag, msg = _split_tag_and_message(inner)
        if not _is_xorg_error(tag, msg):
            return None
        return {
            "original_line": line,
            "severity": severity,
            "count": count,
            "message": msg,
            "host": None,  # хоста в таком формате может не быть
        }

    m = RE_STANDARD.search(line)
    if m:
        severity = int(m.group(1)) & 7
        host = m.group(3)
        full_message = m.group(4).strip()
        tag, msg = _split_tag_and_message(full_message)
        if not _is_xorg_error(tag, msg):
            return None
        return {
            "original_line": line,
            "severity": severity,
            "count": 1,
            "message": msg,
            "host": host,
        }

    return None
