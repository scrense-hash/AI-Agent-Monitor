#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
web_monitor.py
Простой Flask-сервер: при каждом запросе читает все записи из analytics.db и рендерит HTML.
Время отображается в целевой таймзоне (TARGET_TIMEZONE_STR).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template_string

# --- ИМПОРТ ВСТРОЕННОГО МОДУЛЯ ДЛЯ РАБОТЫ С ЧАСОВЫМИ ПОЯСАМИ ---
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import agent_daemon


# --- КОНФИГУРАЦИЯ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(SCRIPT_DIR, "analytics.db"))
# Время обновления
REFRESH_SEC = int(os.getenv("REFRESH_SEC", "5"))
# Целевая таймзона
TARGET_TIMEZONE_STR = os.getenv("TZ", "Europe/Moscow")


app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Rsyslog AI Monitor</title>
  <meta http-equiv="refresh" content="{{ refresh_sec }}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    /* Стили остаются без изменений */
    html, body { margin: 0; padding: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; background: #121212; color: #00FF00; }
    .wrap { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: 24px; margin: 0 0 8px; text-shadow: 0 0 5px rgba(0,255,0,0.5); }
    .meta { color: #00A000; font-size: 13px; margin-bottom: 24px; }
    footer { margin: 24px 0 8px; color: #006000; font-size: 12px; text-align: center; }
    a { color: #50FF50; }
    .incident { padding: 12px 16px; margin: 16px 0; background: #1A1A1A; border-left: 5px solid #00FF00; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
    .incident p { margin: 6px 0; line-height: 1.4; }
    code { background: #000; padding: 2px 5px; border-radius: 3px; border: 1px solid #005000; word-break: break-all; }
    .incident p strong { color: #FFFFFF; }
    .incident.critical { background: #3B0000; border-left-color: #FF4D4F; color: #FFCFCF; }
    .incident.critical p strong { color: #FFFFFF; }
    .incident.critical code { background: #1F0000; border-color:#800000;}
    .incident.error { background: #402300; border-left-color: #FA8C16; color: #FFD59A; }
    .incident.error p strong { color: #FFFFFF; }
    .incident.error code { background: #261400; border-color:#7D450B;}
    .incident.warning { background: #403100; border-left-color: #FAAD14; color: #FFF2B8; }
    .incident.warning p strong { color: #FFFFFF; }
    .incident.warning code { background: #261D00; border-color:#7D580B;}
    .empty { padding: 32px; border: 2px dashed #005000; border-radius: 8px; text-align: center; color: #00A000; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Монитор инцидентов AI-Агента</h1>
  <div class="meta">
    Автообновление каждые <strong>{{ refresh_sec }} сек.</strong>&nbsp; | &nbsp;Всего инцидентов: <strong>{{ rows|length }}</strong>&nbsp; | &nbsp;Время: <strong>{{ timezone }}</strong>
  </div>

  {% if not rows %}
    <div class="empty">Ожидание событий...</div>
  {% else %}
    {% for r in rows %}
      <div class="incident {{ r.css_class }}">
        <p><strong>Инцидент:</strong> {{ r.summary }}</p>
        <p><strong>Хост:</strong> {{ r.last_host or "unknown" }} ({{ r.last_ip or "n/a" }})</p>
        <p><strong>Повторения:</strong> {{ r.count }} раз</p>
        <p><strong>Впервые замечен:</strong> {{ r.first_seen }}</p>
        <p><strong>Последний раз:</strong> {{ r.last_seen }}</p>
        <p><strong>Последний лог:</strong> <code>{{ r.last_log }}</code></p>
      </div>
    {% endfor %}
  {% endif %}

  <footer>© {{ year }} · AI Agent Monitor</footer>
</div>
</body>
</html>
"""


def severity_css_class(sev: int) -> str:
    if sev in (0, 1, 2):
        return "critical"
    if sev == 3:
        return "error"
    if sev == 4:
        return "warning"
    return ""


# --- МОДИФИЦИРОВАННАЯ ФУНКЦИЯ ---
def fetch_rows(db_path: str):
    if not os.path.exists(db_path):
        return []
    conn = None
    try:
        # Устанавливаем целевой часовой пояс. UTC по умолчанию, если указан неверный.
        try:
            target_tz = ZoneInfo(TARGET_TIMEZONE_STR)
        except ZoneInfoNotFoundError:
            app.logger.error(f"Неверный часовой пояс '{TARGET_TIMEZONE_STR}'. Используется UTC.")
            target_tz = ZoneInfo("UTC")

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM incidents ORDER BY last_seen DESC")

        res = []
        for row in cur.fetchall():
            # Преобразуем UTC время из строки в объект datetime (наивный)
            first_seen_naive = datetime.strptime(row["first_seen"], "%Y-%m-%d %H:%M:%S")
            last_seen_naive = datetime.strptime(row["last_seen"], "%Y-%m-%d %H:%M:%S")

            # Делаем его "осведомленным" о том, что это UTC
            first_seen_utc = first_seen_naive.replace(tzinfo=ZoneInfo("UTC"))
            last_seen_utc = last_seen_naive.replace(tzinfo=ZoneInfo("UTC"))

            # Конвертируем в целевой часовой пояс
            first_seen_local = first_seen_utc.astimezone(target_tz)
            last_seen_local = last_seen_utc.astimezone(target_tz)

            res.append({
                "summary": row["summary"],
                "severity": row["severity"],
                "count": row["count"],
                # Отдаем в шаблон уже отформатированную строку с локальным временем
                "first_seen": first_seen_local.strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": last_seen_local.strftime("%Y-%m-%d %H:%M:%S"),
                "last_log": row["last_log"],
                "last_host": row["last_host"],
                "last_ip": row["last_ip"],
                "css_class": severity_css_class(row["severity"]),
            })
        return res
    except sqlite3.OperationalError as e:
        app.logger.warning(f"Не удалось прочитать БД (возможно, заблокирована): {e}")
    finally:
        if conn:
            conn.close()


@app.get("/")
def index():
    rows = fetch_rows(DB_PATH)
    # Передаем имя таймзоны в шаблон
    return render_template_string(
        TEMPLATE,
        rows=rows,
        db_path=DB_PATH,
        year=datetime.now(ZoneInfo("UTC")).year,
        refresh_sec=REFRESH_SEC,
        timezone=TARGET_TIMEZONE_STR
    )


@app.get("/health")
def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_PATH)}


if __name__ == "__main__":
    # Для Python < 3.9 установите backports.zoneinfo: pip install backports.zoneinfo

    # Если переменная окружения WEB_PORT не задана, используем порт по умолчанию
    bind = os.getenv("BIND_INTERFACE", agent_daemon.DEFAULT_BIND_INTERFACE)
    # Если переменная окружения WEB_PORT не задана, используем порт по умолчанию
    port = int(os.getenv("WEB_PORT", str(agent_daemon.DEFAULT_WEB_PORT)))
    try:
        from waitress import serve
        serve(app, host=bind, port=port)
    except ImportError:
        print("Waitress не установлен. Запуск через dev-сервер Flask. Для production установите: pip install waitress")
        app.run(host=bind, port=port, debug=False)
