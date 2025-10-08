# plugins/syslog_default.py
# -*- coding: utf-8 -*-
"""
Default плагин для парсинга стандартных rsyslog-строк (single и repeat).

Возвращаемый словарь: {"original_line", "severity", "count", "message", "host"}

PRIORITY: 1000 (низкий, fallback после specialized).
"""

import re
from typing import Optional, Dict

PRIORITY = 1000

# RE для single: <PRI>TIMESTAMP HOST TAG: MESSAGE
RE_STANDARD = re.compile(
    r"<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+([^\s:]+)\s+([^:]+):\s*(.*)"
)

# RE для repeat: <PRI>... message repeated N times: [MESSAGE]
RE_REPEAT = re.compile(
    r"<(\d+)>\s*(?:\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})?.*?message repeated\s+(\d+)\s+times:\s*\[\s*(.*)\s*\]",
    re.IGNORECASE,
)


def can_handle(line: str) -> bool:
    """Single или repeat."""
    return bool(RE_STANDARD.search(line)) or bool(RE_REPEAT.search(line))


def parse(line: str) -> Optional[Dict]:
    """Парсит single или repeat. Для repeat — извлекает host из original_line."""
    # Сначала пробуем repeat
    m_repeat = RE_REPEAT.search(line)
    if m_repeat:
        pri = int(m_repeat.group(1))
        severity = pri % 8  # 0-7 (PRI % 8 = severity)
        count = int(m_repeat.group(2))
        raw_message = m_repeat.group(3).strip()
        message = _clean_after_colon(raw_message)

        # ИЗВЛЕКАЕМ HOST ИЗ ORIGINAL LINE (используем RE_STANDARD на части строки)
        m_host = RE_STANDARD.search(line)
        host = m_host.group(3) if m_host else None  # group(3): HOST (e.g., admin-pc)

        # Fallback: Если нет host — ищем простой паттерн HOST (after TIMESTAMP)
        if not host:
            timestamp_match = re.search(r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+([^\s:]+)", line)
            host = timestamp_match.group(2) if timestamp_match else "unknown"

        return {
            "original_line": line,
            "severity": severity,
            "count": count,
            "message": message,
            "host": host or "unknown",  # Гарантируем str
        }

    # Single лог
    m_standard = RE_STANDARD.search(line)
    if m_standard:
        pri = int(m_standard.group(1))
        severity = pri % 8
        host = m_standard.group(3)  # HOST
        full_message = m_standard.group(5).strip()  # После TAG:
        message = _clean_after_colon(full_message)

        return {
            "original_line": line,
            "severity": severity,
            "count": 1,
            "message": message,
            "host": host or "unknown",
        }

    # Fallback: Если ничего — пропуск или minimal parse
    pri_match = re.match(r'<(\d+)>', line)
    if pri_match:
        severity = int(pri_match.group(1)) % 8
        # Message = rest, host="unknown"
        message = line.split(' ', 3)[-1] if len(line.split(' ')) > 3 else line
        return {
            "original_line": line,
            "severity": severity,
            "count": 1,
            "message": _clean_after_colon(message),
            "host": "unknown",
        }

    return None


def _clean_after_colon(text: str) -> str:
    """Часть после : , если есть."""
    parts = text.split(":", 1)
    return parts[1].strip() if len(parts) > 1 else text
