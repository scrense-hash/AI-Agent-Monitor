#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_daemon.py (REFACTOURED VERSION)
Агент слушает UDP (rsyslog), извлекает "summary" через LLM-оркестратор и сохраняет инциденты в SQLite.
Парсинг логов через систему плагинов (plugins/).
LLM-логика вынесена в модуль llm/.

Структура:
- llm/base_provider.py - базовый класс провайдера
- llm/local_provider.py - локальная модель (llama.cpp)
- llm/cloud_provider.py - облачный провайдер (OpenAI совместимый)
- llm/orchestrator.py - оркестратор с fallback
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sqlite3
import sys
import time
import subprocess
import atexit
import importlib.util
import glob
import re
import ipaddress
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, List, Any

# Импорт LLM модулей
import dotenv  # NOTE: There might be a Pylance issue with resolving the dotenv import even after installing the package.
from llm.local_provider import LocalProvider
from llm.cloud_provider import CloudProvider
from llm.orchestrator import LLMOrchestrator


# -------- Константы --------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "analytics.db")
DEFAULT_DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "debug.log")

# Привязка интерфейса общая для rsyslog и веб-сервера
DEFAULT_BIND_INTERFACE = "0.0.0.0"

# Порт и буфер для rsyslog
DEFAULT_RSYSLOG_PORT = 1514
DEFAULT_RSYSLOG_RECV_BUF = 8192

# Фильтр IP-адресов для приёма логов (строка с адресами/подсетями через запятую; пустая строка = без фильтра)
DEFAULT_HOST_FILTER = ""


def parse_host_filter(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]

# Порт для встроенного веб-сервера мониторинга
DEFAULT_WEB_PORT = 8000

# Уровень серьезности (severity) для обработки.
# Соответствует стандартной шкале syslog:
# 0 - Emergency (система не работоспособна)
# 1 - Alert (требуется немедленное внимание)
# 2 - Critical (критическая ошибка)
# 3 - Error (ошибка, требующая вмешательства)
# 4 - Warning (предупреждение о потенциальной проблеме)
# 5 - Notice (значимое событие, но не ошибка)
# 6 - Informational (информационные сообщения)
# 7 - Debug (отладочная информация)
DEFAULT_MAX_SEVERITY_TO_PROCESS = 4

# Очистка БД
DEFAULT_CLEAN_ON_START = True
DEFAULT_CLEAN_INTERVAL = 60 * 60 * 24 * 7  # 7 дней

# Настройки для локальной модели
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "phi-3-mini-4k-instruct-q4_k_m.gguf")
DEFAULT_N_CTX = 4096
DEFAULT_N_THREADS = -1
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0

# Облачный API (OpenAI совместимый)
DEFAULT_CLOUD_MODEL = "minimax/minimax-m2:free"
DEFAULT_CLOUD_API_BASE_URL = "https://openrouter.ai/api/v1"

# Настройки HTTP прокси для облачных API
DEFAULT_PROXY_USE = False
DEFAULT_PROXY_HOST = ""
DEFAULT_PROXY_PORT = 0


# -------- Конфигурация --------

@dataclass
class Config:
    db_path: str = DEFAULT_DB_PATH
    debug_log_path: str = DEFAULT_DEBUG_LOG_PATH
    host: str = DEFAULT_BIND_INTERFACE
    port: int = DEFAULT_RSYSLOG_PORT
    recv_buf: int = DEFAULT_RSYSLOG_RECV_BUF
    max_severity: int = DEFAULT_MAX_SEVERITY_TO_PROCESS
    host_filter: List[str] = field(default_factory=list)

    # Локальная модель
    model_path: str = DEFAULT_MODEL_PATH
    n_ctx: int = DEFAULT_N_CTX
    n_threads: int = DEFAULT_N_THREADS
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    # Облачный API
    cloud_api_key: Optional[str] = None
    cloud_api_base_url: str = DEFAULT_CLOUD_API_BASE_URL
    cloud_model: str = DEFAULT_CLOUD_MODEL
    enable_cloud: bool = False
    
    # HTTP прокси для облачных API
    proxy_host: str = DEFAULT_PROXY_HOST
    proxy_port: int = DEFAULT_PROXY_PORT
    proxy_use: bool = DEFAULT_PROXY_USE

    # Очистка БД
    clean_interval: int = DEFAULT_CLEAN_INTERVAL
    clean_on_start: bool = DEFAULT_CLEAN_ON_START

    @staticmethod
    def from_env_and_args() -> "Config":
        dotenv.load_dotenv()
        p = argparse.ArgumentParser(description="Rsyslog AI-Agent (Refactored with LLM modules)")
        p.add_argument("--db", default=os.getenv("DB_PATH", DEFAULT_DB_PATH))
        p.add_argument("--debug-log", default=os.getenv("DEBUG_LOG_PATH", DEFAULT_DEBUG_LOG_PATH))
        p.add_argument("--bind-interface", default=os.getenv("BIND_INTERFACE", DEFAULT_BIND_INTERFACE))
        p.add_argument("--rsyslog-port", type=int, default=int(os.getenv("RSYSLOG_PORT", DEFAULT_RSYSLOG_PORT)))
        p.add_argument("--web-port", type=int, default=int(os.getenv("WEB_PORT", DEFAULT_WEB_PORT)))
        p.add_argument("--rsyslog-recv-buf", type=int, default=int(os.getenv("RSYSLOG_RECV_BUF", DEFAULT_RSYSLOG_RECV_BUF)))
        p.add_argument("--max-sev", type=int, default=int(os.getenv("MAX_SEVERITY_TO_PROCESS", DEFAULT_MAX_SEVERITY_TO_PROCESS)))
        host_filter_env = os.getenv("HOST_FILTER")
        if host_filter_env is None or host_filter_env.strip() == "":
            host_filter_env = os.getenv("DEFAULT_HOST_FILTER", DEFAULT_HOST_FILTER)
        p.add_argument("--host-filter", default=host_filter_env, help="Список IP/подсетей для приёма логов (через запятую)")

        # Локальная модель
        p.add_argument("--model-path", default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        p.add_argument("--n-ctx", type=int, default=int(os.getenv("N_CTX", DEFAULT_N_CTX)))
        p.add_argument("--n-threads", type=int, default=int(os.getenv("N_THREADS", DEFAULT_N_THREADS)))
        p.add_argument("--n-gpu-layers", type=int, default=int(os.getenv("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)))
        p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)))
        p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)))

        # Облачный API
        p.add_argument("--cloud-api-key", default=os.getenv("CLOUD_API_KEY", os.getenv("OPENAI_API_KEY")))
        p.add_argument("--cloud-api-base-url", default=os.getenv("CLOUD_API_BASE_URL", DEFAULT_CLOUD_API_BASE_URL))
        p.add_argument("--cloud-model", default=os.getenv("CLOUD_MODEL", DEFAULT_CLOUD_MODEL))
        p.add_argument("--enable-cloud", action="store_true", default=os.getenv("ENABLE_CLOUD", "").lower() in ("1", "true", "yes"))
        
        # HTTP прокси
        proxy_host_env = os.getenv("PROXY_HOST")
        if proxy_host_env is None or proxy_host_env.strip() == "":
            proxy_host_env = os.getenv("DEFAULT_PROXY_HOST", DEFAULT_PROXY_HOST)

        proxy_port_env = os.getenv("PROXY_PORT")
        if proxy_port_env is None or str(proxy_port_env).strip() == "":
            proxy_port_env = os.getenv("DEFAULT_PROXY_PORT")
        if proxy_port_env is None or str(proxy_port_env).strip() == "":
            proxy_port_env = str(DEFAULT_PROXY_PORT)

        env_proxy_use_raw = os.getenv("PROXY_USE")
        if env_proxy_use_raw is None or str(env_proxy_use_raw).strip() == "":
            env_proxy_use_raw = os.getenv("DEFAULT_PROXY_USE", str(DEFAULT_PROXY_USE))
        env_proxy_use = str(env_proxy_use_raw).strip().lower()
        proxy_use_default = env_proxy_use in ("1", "true", "yes", "on")

        p.add_argument("--proxy-host", default=proxy_host_env)
        p.add_argument("--proxy-port", type=int, default=int(proxy_port_env))
        p.add_argument("--proxy-use", dest="proxy_use", action="store_true", help="Включить HTTP прокси для облачных API")
        p.add_argument("--no-proxy-use", dest="proxy_use", action="store_false", help="Отключить HTTP прокси для облачных API")
        p.set_defaults(proxy_use=proxy_use_default)

        # Очистка
        p.add_argument("--clean-interval", type=int, default=int(os.getenv("CLEAN_INTERVAL", DEFAULT_CLEAN_INTERVAL)))
        p.add_argument("--clean-on-start", default=os.getenv("CLEAN_ON_START", str(DEFAULT_CLEAN_ON_START)))

        a = p.parse_args()
        return Config(
            db_path=a.db,
            debug_log_path=a.debug_log,
            host=a.bind_interface,
            port=a.rsyslog_port,
            recv_buf=a.rsyslog_recv_buf,
            max_severity=a.max_sev,
            host_filter=parse_host_filter(a.host_filter),
            model_path=a.model_path,
            n_ctx=a.n_ctx,
            n_threads=a.n_threads,
            n_gpu_layers=a.n_gpu_layers,
            max_tokens=a.max_tokens,
            temperature=a.temperature,
            cloud_api_key=a.cloud_api_key,
            cloud_api_base_url=a.cloud_api_base_url,
            cloud_model=a.cloud_model,
            enable_cloud=a.enable_cloud,
            proxy_host=a.proxy_host,
            proxy_port=a.proxy_port,
            proxy_use=a.proxy_use,
            clean_interval=a.clean_interval,
            clean_on_start=str(a.clean_on_start).strip().lower() in ("1", "true", "yes", "on"),
        )


# -------- Логирование --------

def setup_logging(debug_log_path: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = RotatingFileHandler(debug_log_path, maxBytes=1 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)


# -------- Утилиты --------

def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def make_summary_key(summary: str, host: str = "") -> str:
    """Создает ключ для агрегации инцидентов по хосту."""
    combined = f"{summary.strip()} || {host or 'unknown'}"
    s = combined.lower()
    s = re.sub(r"\s+", " ", s)
    return s


# -------- База данных --------

def connect_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.Connection(path, isolation_level=None, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            summary     TEXT    NOT NULL,
            summary_key TEXT    NOT NULL UNIQUE,
            severity    INTEGER NOT NULL,
            count       INTEGER NOT NULL DEFAULT 0,
            first_seen  TEXT    NOT NULL,
            last_seen   TEXT    NOT NULL,
            last_log    TEXT    NOT NULL,
            last_host   TEXT,
            last_ip     TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_last_seen ON incidents(last_seen DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);")


def clean_old_incidents(conn: sqlite3.Connection, *, max_age_seconds: int) -> int:
    if max_age_seconds <= 0:
        return 0
    cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    logging.debug(f"Удаление инцидентов старше: {cutoff_str}")
    cur = conn.execute("DELETE FROM incidents WHERE last_seen < ?", (cutoff_str,))
    return cur.rowcount if cur is not None else 0


def clean_all_incidents(conn: sqlite3.Connection) -> int:
    logging.debug("Принудительная очистка ВСЕХ инцидентов")
    cur = conn.execute("DELETE FROM incidents;")
    return cur.rowcount if cur is not None else 0


def upsert_incident(
    conn: sqlite3.Connection,
    *,
    summary: str,
    severity: int,
    count: int,
    last_log: str,
    last_host: Optional[str],
    last_ip: str,
    host_for_key: str,
) -> None:
    ts = now_ts()
    summary_key = make_summary_key(summary, host_for_key)
    last_host_str = last_host or "unknown"

    logging.debug(f"Upsert: {summary_key} (severity={severity}, count={count}, host={host_for_key})")

    conn.execute(
        """
        INSERT INTO incidents (summary, summary_key, severity, count, first_seen, last_seen, last_log, last_host, last_ip)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(summary_key) DO UPDATE SET
            count     = incidents.count + excluded.count,
            severity  = MIN(incidents.severity, excluded.severity),
            last_seen = excluded.last_seen,
            last_log  = excluded.last_log,
            last_host = COALESCE(excluded.last_host, incidents.last_host),
            last_ip   = excluded.last_ip
        ;
        """,
        (summary, summary_key, severity, count, ts, ts, last_log, last_host_str, last_ip),
    )


# -------- Загрузка плагинов --------

def load_plugins(plugins_dir: str) -> List[Any]:
    loaded = []
    if not os.path.isdir(plugins_dir):
        logging.warning(f"Папка с плагинами не найдена: {plugins_dir}")
        return loaded

    pattern = os.path.join(plugins_dir, "*.py")
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("_"):
            continue

        mod_name = f"plugins.{name}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
            else:
                continue

            if not hasattr(module, "can_handle") or not hasattr(module, "parse"):
                logging.warning(f"Модуль {name} пропущен: нет can_handle/parse")
                continue

            prio = getattr(module, "PRIORITY", 100)
            loaded.append((prio, name, module))
            logging.info(f"Загружен плагин: {name} (PRIORITY={prio})")

        except Exception as e:
            logging.exception(f"Ошибка загрузки плагина {name}: {e}")

    loaded.sort(key=lambda t: t[0])
    return [m for _prio, _name, m in loaded]


# -------- Основной агент --------

class Agent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._stop = False
        self.host_filters = self._compile_host_filters(cfg.host_filter)

        # Инициализация LLM-оркестратора
        logging.info("=" * 60)
        logging.info("Инициализация LLM-провайдеров...")
        logging.info("=" * 60)

        providers = []

        if not cfg.proxy_use and cfg.proxy_host and cfg.proxy_port:
            proxy_url = f"http://{cfg.proxy_host}:{cfg.proxy_port}"
            for env_var in ("HTTP_PROXY", "HTTPS_PROXY"):
                if os.environ.get(env_var) == proxy_url:
                    os.environ.pop(env_var, None)
                    logging.info(f"Сброшена переменная окружения {env_var} (HTTP прокси отключен)")

        # 2. Облачный провайдер (приоритет 2, опционально)
        if cfg.enable_cloud:
            try:
                cloud = CloudProvider(
                    api_key=cfg.cloud_api_key,
                    base_url=cfg.cloud_api_base_url,
                    model_name=cfg.cloud_model,
                    proxy_host=cfg.proxy_host if cfg.proxy_use else None,
                    proxy_port=cfg.proxy_port if cfg.proxy_use else None
                )
                if cloud.is_available:
                    providers.append(cloud)
            except Exception as e:
                logging.error(f"Не удалось инициализировать CloudProvider: {e}")


        # 1. Локальная модель (приоритет 1)
        try:
            local = LocalProvider(
                model_path=cfg.model_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_gpu_layers=cfg.n_gpu_layers,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature
            )
            if local.is_available:
                providers.append(local)
        except Exception as e:
            logging.error(f"Не удалось инициализировать LocalProvider: {e}")

        
        # Создаем оркестратор
        self.llm_orchestrator = LLMOrchestrator(providers)

        if not providers:
            logging.critical("НЕТ ДОСТУПНЫХ LLM-ПРОВАЙДЕРОВ! Будет использоваться только rule-based fallback.")

        logging.info("=" * 60)

        # База данных
        self.conn = connect_db(cfg.db_path)
        init_db(self.conn)

        # Плагины
        self.plugins = load_plugins(os.path.join(SCRIPT_DIR, "plugins"))
        if not self.plugins:
            logging.warning("Не загружено ни одного плагина! Добавьте в папку plugins/.")

    def stop(self, *_args) -> None:
        logging.info("Остановка агента...")
        self._stop = True

    def should_process(self, severity: int) -> bool:
        return severity <= self.cfg.max_severity

    def _dispatch_parse(self, line: str) -> Optional[Dict]:
        for plugin in self.plugins:
            try:
                if plugin.can_handle(line):
                    parsed = plugin.parse(line)
                    if parsed:
                        required = {"original_line", "severity", "count", "message", "host"}
                        missing = required - set(parsed.keys())
                        if missing:
                            logging.warning(f"Плагин {getattr(plugin, '__name__', plugin)}: missing {missing}")
                            continue
                        logging.debug(f"Плагин {getattr(plugin, '__name__', plugin)}: OK")
                        return parsed
            except Exception as e:
                logging.exception(f"Ошибка плагина {getattr(plugin, '__name__', plugin)}: {e}")
        logging.debug("Нет подходящего плагина — пропуск.")
        return None

    def _compile_host_filters(self, filters: List[str]) -> List[ipaddress._BaseNetwork]:
        compiled: List[ipaddress._BaseNetwork] = []
        for raw in filters:
            try:
                network = ipaddress.ip_network(raw, strict=False)
                compiled.append(network)
            except ValueError:
                logging.warning(f"Некорректный фильтр IP '{raw}' — пропуск")
        if filters and not compiled:
            logging.warning("Фильтр IP-адресов задан, но не удалось разобрать ни одного значения")
        return compiled

    def _is_ip_allowed(self, ip: str) -> bool:
        if not self.host_filters:
            return True
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            logging.warning(f"Получен некорректный IP-адрес '{ip}' — сообщение пропущено")
            return False
        return any(ip_obj in network for network in self.host_filters)

    def clean_db_on_start(self) -> None:
        try:
            if self.cfg.clean_on_start:
                removed = clean_all_incidents(self.conn)
                logging.info(f"Принудительная очистка ВСЕХ инцидентов при старте: удалено {removed}")
            else:
                removed = clean_old_incidents(self.conn, max_age_seconds=self.cfg.clean_interval)
                logging.info(f"Очистка старых инцидентов при старте: удалено {removed}")
        except Exception as e:
            logging.exception(f"Ошибка очистки БД при старте: {e}")

    def serve(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.cfg.host, self.cfg.port))
            logging.info(f"Агент запущен на {self.cfg.host}:{self.cfg.port}, БД: {self.cfg.db_path}")
            logging.info(f"Уровень логов: {self.cfg.max_severity} (0..{self.cfg.max_severity})")
            logging.info(f"Интервал очистки: {self.cfg.clean_interval} сек ({self.cfg.clean_interval // (60 * 60 * 24)} дней)")
        except Exception as e:
            logging.critical(f"Не удалось запустить сервер: {e}")
            sys.exit(1)

        self.clean_db_on_start()
        last_clean_ts = time.time()
        last_stats_ts = time.time()

        while not self._stop:
            try:
                sock.settimeout(1.0)
                data, (src_ip, src_port) = sock.recvfrom(self.cfg.recv_buf)
            except socket.timeout:
                now = time.time()

                # Периодическая очистка
                if now - last_clean_ts >= 60.0:
                    try:
                        removed = clean_old_incidents(self.conn, max_age_seconds=self.cfg.clean_interval)
                        if removed:
                            logging.info(f"Периодическая очистка: удалено {removed}")
                    except Exception as e:
                        logging.exception(f"Ошибка периодической очистки: {e}")
                    last_clean_ts = now

                # Статистика LLM (каждые 5 минут)
                if now - last_stats_ts >= 300.0:
                    self.llm_orchestrator.log_stats()
                    last_stats_ts = now

                continue
            except Exception as e:
                if not self._stop:
                    logging.error(f"Ошибка чтения сокета: {e}")
                continue

            try:
                line = data.decode("utf-8", errors="ignore").strip()
                if not self._is_ip_allowed(src_ip):
                    logging.debug(f"Пропуск лога от {src_ip} — не входит в разрешённый фильтр IP")
                    continue
                logging.info(f"Получено сообщение от rsyslog: {src_ip}:{src_port} (длина: {len(line)})")
                logging.debug(f"Содержимое: {line}")

                parsed = self._dispatch_parse(line)
                if not parsed:
                    logging.debug("Пропуск — нет подходящего плагина")
                    continue

                logging.debug(f"Парсинг успешен: {parsed}")

                host = parsed.get("host") or "unknown"
                ip = src_ip

                if not self.should_process(parsed["severity"]):
                    if parsed["count"] > 1:
                        logging.warning(f"Пропуск repeat (sev={parsed['severity']} > max={self.cfg.max_severity}) от {host}, count={parsed['count']}")
                    else:
                        logging.debug(f"Пропуск по severity (sev={parsed['severity']} > max={self.cfg.max_severity})")
                    continue

                logging.info(f"ВАЖНЫЙ ЛОГ (sev={parsed['severity']}, x{parsed['count']}) от {host} ({ip})")

                # Получаем summary через оркестратор
                summary = self.llm_orchestrator.get_summary(parsed["message"])

                upsert_incident(
                    self.conn,
                    summary=summary,
                    severity=parsed["severity"],
                    count=parsed["count"],
                    last_log=parsed["original_line"],
                    last_host=host,
                    last_ip=ip,
                    host_for_key=host,
                )

                logging.info("Инцидент успешно сохранён в БД")

            except Exception as e:
                logging.exception(f"Ошибка обработки сообщения: {e}")

        try:
            self.conn.close()
            sock.close()
        except Exception:
            pass

        # Финальная статистика
        logging.info("=" * 60)
        logging.info("ФИНАЛЬНАЯ СТАТИСТИКА LLM:")
        self.llm_orchestrator.log_stats()
        logging.info("=" * 60)
        logging.info("Агент остановлен.")


def check_dependencies():
    """Проверка зависимостей Python"""
    required_packages = {
        "llama_cpp": "llama-cpp-python",
        "json_repair": "json-repair",
        "langid": "langid",
        "langdetect": "langdetect",
        "requests": "requests",
        "flask": "flask",
        "zoneinfo": "backports.zoneinfo",  # For Python < 3.9
        "waitress": "waitress",
        "dotenv": "python-dotenv",
    }

    missing_packages = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        logging.critical("Некоторые необходимые Python пакеты не установлены:")
        for package_name in missing_packages:
            logging.critical(f"  - {package_name}")
        logging.critical("Пожалуйста, установите их с помощью: pip install {}".format(" ".join(missing_packages)))
        sys.exit(1)

def main() -> None:

    # Check dependencies
    check_dependencies()

    cfg = Config.from_env_and_args()
    setup_logging(cfg.debug_log_path)

    logging.info("=" * 60)
    logging.info("RSYSLOG AI-AGENT - REFACTORED VERSION")
    logging.info("=" * 60)
    logging.info(f"База данных: {cfg.db_path}")
    logging.info(f"Лог отладки: {cfg.debug_log_path}")
    logging.info(f"Max severity: {cfg.max_severity}")
    logging.info(f"Фильтр IP: {', '.join(cfg.host_filter) if cfg.host_filter else 'нет'}")
    logging.info(f"Интервал очистки: {cfg.clean_interval} сек")
    logging.info(f"Принудительная очистка ВСЕХ инцидентов при старте: {cfg.clean_on_start}")
    logging.info("")
    logging.info("КОНФИГУРАЦИЯ LLM:")
    logging.info(f"  Локальная модель: {cfg.model_path}")
    logging.info(f"  Контекст: {cfg.n_ctx}, Потоки: {cfg.n_threads}, GPU слои: {cfg.n_gpu_layers}")
    if cfg.enable_cloud:
        logging.info(f"  Облачный API: ВКЛЮЧЕН (модель: {cfg.cloud_model})")
        logging.info(f"  Базовый URL: {cfg.cloud_api_base_url}")
        if cfg.proxy_use and cfg.proxy_host and cfg.proxy_port:
            logging.info(f"  HTTP прокси: {cfg.proxy_host}:{cfg.proxy_port}")
        elif not cfg.proxy_use:
            logging.info("  HTTP прокси: отключен")
        else:
            logging.warning("  HTTP прокси: включен, но не указаны host/port")
    else:
        logging.info("  Облачный API: ВЫКЛЮЧЕН")
    logging.info("=" * 60)

    # --- Запуск web_monitor ---
    monitor_proc = None
    try:
        env = os.environ.copy()

        web_monitor_path = os.path.join(SCRIPT_DIR, "web_monitor.py")
        if os.path.exists(web_monitor_path):
            logging.info("Запуск web_monitor")
            monitor_proc = subprocess.Popen([sys.executable, web_monitor_path], env=env)

            def _stop_monitor() -> None:
                try:
                    if monitor_proc and monitor_proc.poll() is None:
                        logging.info("Остановка web_monitor...")
                        monitor_proc.terminate()
                except Exception:
                    pass

            atexit.register(_stop_monitor)
        else:
            logging.warning("web_monitor.py не найден.")

        # --- Запуск агента ---
        Agent(cfg).serve()

    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        sys.exit(1)
    finally:
        try:
            if monitor_proc and monitor_proc.poll() is None:
                monitor_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
