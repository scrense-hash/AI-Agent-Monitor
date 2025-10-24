# llm/local_provider.py
# -*- coding: utf-8 -*-
"""
Провайдер для локальной LLM (llama.cpp).
Загружает и использует локальную модель для генерации summary.
"""

import json
import logging
import os
import re
import time
from typing import Optional, Dict

from .base_provider import BaseLLMProvider

# Опциональные зависимости
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    def repair_json(s: str) -> str:
        return s

try:
    import langid
    LANGID_AVAILABLE = True
except Exception:
    LANGID_AVAILABLE = False

try:
    from langdetect import detect as ld_detect, DetectorFactory as LD_DetectorFactory
    LD_DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False


# Промпт для модели
SYSTEM_PROMPT = """
Ты — Анализатор сообщений логов. Твоя задача — понять суть чистого сообщения из лога (без даты и хоста) и вернуть ТОЛЬКО один валидный JSON-объект. Никаких объяснений, дополнительного текста, кодовых блоков или markdown. Остановись сразу после JSON.

Формат должен быть ВЫПОЛНИТЕЛЬНО таким:
{{ "summary": "Полное описание на русском (максимум 30 слов)" }}

ПРИМЕРЫ:
Вход: read failure
Выход: {{"summary": "Ошибка чтения данных"}}

Вход: kernel panic
Выход: {{"summary": "Паника ядра: критический сбой системы"}}

Вход: I/O error, dev sda, sector 12345
Выход: {{"summary": "Ошибка ввода-вывода на диске sda, сектор 12345"}}

Вход: connection refused from 192.168.1.1
Выход: {{"summary": "Отказ в соединении по сети от IP 192.168.1.1"}}

Сообщение для анализа: {message}
""".strip()

# Регулярное выражение для поиска JSON
RE_FIRST_JSON = re.compile(r"\{[^{}]*?(?:\{[^{}]*?\}[^{}]*?)*?\}", re.DOTALL)


class LocalProvider(BaseLLMProvider):
    """
    Провайдер для локальной LLM через llama.cpp.
    Поддерживает кэширование, retry для русского языка и rule-based fallback.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = -1,
                 n_gpu_layers: int = 0, max_tokens: int = 256, temperature: float = 0.0):
        """
        Инициализирует локальный провайдер.

        Args:
            model_path: Путь к файлу модели .gguf
            n_ctx: Размер контекста
            n_threads: Количество потоков (-1 для автоопределения)
            n_gpu_layers: Количество слоев на GPU
            max_tokens: Максимальное количество токенов для генерации
            temperature: Температура сэмплирования
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llama = None
        self._is_ready = False

        # Управляемый кэш
        self._summary_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if not LLAMA_CPP_AVAILABLE:
            logging.error("llama-cpp-python не установлен. LocalProvider недоступен.")
            return

        if not os.path.exists(model_path):
            logging.error(f"Модель не найдена: {model_path}. LocalProvider недоступен.")
            return

        try:
            logging.info(f"[LocalProvider] Загрузка модели: {model_path}")
            self.llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads if n_threads > 0 else None,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._is_ready = True
            logging.info("[LocalProvider] Модель успешно загружена")
        except Exception as e:
            logging.error(f"[LocalProvider] Ошибка загрузки модели: {e}")
            self._is_ready = False

    @property
    def name(self) -> str:
        return "LocalLlama"

    @property
    def is_available(self) -> bool:
        return self._is_ready and self.llama is not None

    def get_cache_info(self) -> dict:
        """Возвращает статистику кэша."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._summary_cache)
        }

    def _get_from_cache(self, message: str) -> Optional[str]:
        """Получает summary из кэша."""
        key = message.strip().rstrip('.')
        if key in self._summary_cache:
            self._cache_hits += 1
            logging.debug(f"[{self.name}] Кэш hit: {key[:50]}")
            return self._summary_cache[key]
        self._cache_misses += 1
        return None

    def _set_cache(self, message: str, summary: str) -> None:
        """Сохраняет summary в кэш."""
        key = message.strip().rstrip('.')
        self._summary_cache[key] = summary

    def _purge_cache_key(self, message: str) -> None:
        """Удаляет ключ из кэша."""
        key = message.strip().rstrip('.')
        if key in self._summary_cache:
            del self._summary_cache[key]

    @staticmethod
    def _extract_first_json(text: str) -> Optional[Dict]:
        """Извлекает первый валидный JSON из текста."""
        t = text.strip()

        # Очистка от markdown и прочего шума
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r'```?\s*$', '', t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r'^(?:Here is|Output:|JSON:)\s*', '', t, flags=re.IGNORECASE | re.MULTILINE)

        # Ограничить до первого }
        if '}' in t:
            t = t.split('}', 1)[0] + '}'

        # Пробуем найти несколько кандидатов
        candidates = RE_FIRST_JSON.findall(t)
        for candidate in candidates[:3]:
            fixed = repair_json(candidate) if JSON_REPAIR_AVAILABLE else candidate
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "summary" in obj and isinstance(obj["summary"], str) and obj["summary"].strip():
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

        # Fallback: весь текст
        try:
            fixed = repair_json(t) if JSON_REPAIR_AVAILABLE else t
            obj = json.loads(fixed)
            if isinstance(obj, dict) and "summary" in obj and isinstance(obj["summary"], str) and obj["summary"].strip():
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    @staticmethod
    def _detect_language_quiet(text: str) -> tuple:
        """Определяет язык текста. Возвращает (lang_code, confidence, source)."""
        t = (text or "").strip()
        if not t:
            return ("und", 0.0, "empty")

        # Пробуем langid (с confidence)
        if LANGID_AVAILABLE:
            try:
                lang, conf = langid.classify(t)
                return (lang or "und", float(conf), "langid")
            except Exception:
                pass

        # Fallback: langdetect (без confidence)
        if LANGDETECT_AVAILABLE:
            try:
                lang = ld_detect(t)
                return (lang or "und", 0.8, "langdetect")
            except Exception:
                pass

        # Эвристика: кириллица vs латиница
        try:
            has_cyrillic = bool(re.search(r"[а-яёА-ЯЁ]", t))
            if has_cyrillic:
                return ("ru", 0.7, "heuristic_cyrillic")

            ascii_text = re.sub(r"[^A-Za-z]", "", t)
            if ascii_text:
                vowels = len(re.findall(r"[AEIOUaeiou]", ascii_text))
                ratio = vowels / max(len(ascii_text), 1)
                if ratio >= 0.25:
                    return ("en", 0.6, "heuristic_ascii_vowels")
            return ("und", 0.3, "heuristic_unknown")
        except Exception:
            return ("und", 0.0, "heuristic_error")

    @staticmethod
    def _detect_english(summary: str) -> bool:
        """Проверяет, является ли текст английским."""
        lang, conf, _src = LocalProvider._detect_language_quiet(summary)
        return lang == "en" and conf >= 0.55

    @staticmethod
    def _fallback_summary(message: str) -> str:
        """Rule-based fallback summary на русском."""
        message_lower = message.lower()
        if any(word in message_lower for word in ["error", "ошибка", "fail", "ошиб"]):
            return f"Ошибка в логе: {message[:50]}"
        elif any(word in message_lower for word in ["panic", "паника", "crash"]):
            return "Паника системы: критический сбой"
        elif any(word in message_lower for word in ["connection", "соединение", "refused"]):
            return "Проблема с сетевым соединением"
        elif "i/o" in message_lower or "input" in message_lower or "output" in message_lower:
            return "Ошибка ввода-вывода"
        return f"Событие в логе: {message[:50]}"

    def _retry_russian(self, previous_summary: str, message: str) -> Optional[str]:
        """Retry с уточнением для перевода на русский."""
        try:
            retry_system = SYSTEM_PROMPT.format(message=message)
        except KeyError as e:
            logging.error(f"[{self.name}] Ошибка форматирования retry_prompt: {e}")
            return None

        retry_prompt = (
            f"{retry_system}\n\n"
            f"Предыдущий вариант (на английском): {previous_summary}. "
            f"Переведи summary ТОЛЬКО на русский и верни JSON без изменений формата. "
            f"Не используй английские слова."
        )
        full_retry = f"<|system|>\n{retry_prompt}<|end|>\n<|assistant|>\n"

        try:
            response = self.llama(full_retry, max_tokens=100, temperature=0.0, stop=["}", "\n"], echo=False)
            raw = response["choices"][0]["text"].strip()
            logging.debug(f"[{self.name}] Retry response: {raw[:200]}")
            obj = self._extract_first_json(raw)
            if obj and "summary" in obj:
                ru_summary = str(obj["summary"]).strip()
                logging.info(f"[{self.name}] Retry RU success: {ru_summary}")
                return ru_summary
        except Exception as e:
            logging.error(f"[{self.name}] Retry error: {e}")
        return None

    def summarize(self, message: str) -> Optional[str]:
        """
        Генерирует summary для сообщения лога.

        Args:
            message: Текст сообщения из лога

        Returns:
            Optional[str]: Summary на русском языке или None при неудаче
        """
        # Проверка кэша
        cached = self._get_from_cache(message)
        if cached is not None:
            return cached

        if not self.is_available:
            logging.error(f"[{self.name}] Провайдер недоступен")
            return None

        clean_message = message.strip().rstrip('.')

        # Форматирование промпта
        try:
            formatted_system = SYSTEM_PROMPT.format(message=clean_message)
        except KeyError as e:
            logging.error(f"[{self.name}] Ошибка форматирования промпта: {e}")
            return None

        user_prompt = f"Сообщение для анализа: {clean_message}"
        full_prompt = (
            f"<|system|>\n{formatted_system}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                response = self.llama(
                    full_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["<|end|>", "\n\n", "}"],
                    echo=False,
                )
                inference_time = time.perf_counter() - start_time
                logging.debug(f"[{self.name}] Inference time (attempt {attempt + 1}): {inference_time:.2f}s")

                raw_text = response["choices"][0]["text"].strip()
                logging.debug(f"[{self.name}] Response: {raw_text[:200]}")

                obj = self._extract_first_json(raw_text)
                if obj and "summary" in obj:
                    summary = str(obj["summary"]).strip()

                    # Проверка языка
                    if self._detect_english(summary) and attempt < max_retries - 1:
                        logging.warning(f"[{self.name}] Summary на английском: {summary}. Retry for RU.")
                        ru_summary = self._retry_russian(summary, clean_message)
                        if ru_summary:
                            self._purge_cache_key(clean_message)
                            self._set_cache(clean_message, ru_summary)
                            return ru_summary
                        continue

                    # Логируем язык
                    det_lang, det_conf, det_src = self._detect_language_quiet(summary)
                    logging.info(f"[{self.name}] Summary language: {det_lang.upper()} ({det_conf:.2f}, {det_src}) | {summary}")

                    self._set_cache(clean_message, summary)
                    return summary

            except Exception as e:
                logging.error(f"[{self.name}] Error on attempt {attempt + 1}: {e}")
                continue

        # Все попытки провалились
        logging.warning(f"[{self.name}] All retries failed for: {message[:100]}")
        return None
