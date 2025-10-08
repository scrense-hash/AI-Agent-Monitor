# llm/google_provider.py
# -*- coding: utf-8 -*-
"""
Провайдер для Google Gemini API.
Использует облачный API для генерации summary.
"""

import json
import logging
import os
import re
from typing import Optional, Dict

from .base_provider import BaseLLMProvider

# Опциональная зависимость
try:
    import google.generativeai as genai
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


# Промпт для Gemini
SYSTEM_INSTRUCTION = """
Ты — Анализатор сообщений системных логов.

Твоя задача — понять суть сообщения из лога и вернуть ТОЛЬКО валидный JSON-объект с кратким описанием проблемы НА РУССКОМ ЯЗЫКЕ.

Формат ответа (строго JSON):
{"summary": "Краткое описание на русском (максимум 30 слов)"}

ВАЖНО:
- Только JSON, никакого дополнительного текста
- Summary ТОЛЬКО на русском языке
- Без markdown, кодовых блоков, объяснений
- Максимум 30 слов

ПРИМЕРЫ:
Вход: "read failure"
Выход: {"summary": "Ошибка чтения данных"}

Вход: "kernel panic"
Выход: {"summary": "Паника ядра: критический сбой системы"}

Вход: "I/O error, dev sda, sector 12345"
Выход: {"summary": "Ошибка ввода-вывода на диске sda, сектор 12345"}
"""


class GoogleProvider(BaseLLMProvider):
    """
    Провайдер для Google Gemini API.
    Использует модель gemini-pro или gemini-1.5-flash для генерации summary.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Инициализирует Google провайдер.

        Args:
            api_key: API ключ Google (если None, берется из GOOGLE_API_KEY)
            model_name: Имя модели (gemini-pro, gemini-1.5-flash и т.д.)
        """
        self.model = None
        self._is_ready = False
        self.model_name = model_name

        # Управляемый кэш
        self._summary_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if not GOOGLE_API_AVAILABLE:
            logging.warning("[GoogleProvider] google-generativeai не установлен. Провайдер недоступен.")
            return

        # Получаем API ключ
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            logging.warning("[GoogleProvider] GOOGLE_API_KEY не установлен. Провайдер недоступен.")
            return

        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_INSTRUCTION
            )
            self._is_ready = True
            logging.info(f"[GoogleProvider] Инициализирован с моделью {model_name}")
        except Exception as e:
            logging.error(f"[GoogleProvider] Ошибка инициализации: {e}")
            self._is_ready = False

    @property
    def name(self) -> str:
        return f"GoogleGemini({self.model_name})"

    @property
    def is_available(self) -> bool:
        return self._is_ready and self.model is not None

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

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict]:
        """
        Извлекает JSON из текста ответа.
        Поддерживает различные форматы ответа от Gemini.
        """
        t = text.strip()

        # Удаляем markdown обертки
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r'```?\s*$', '', t, flags=re.IGNORECASE | re.MULTILINE)

        # Пробуем найти JSON блок
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', t, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict) and "summary" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

        # Пробуем распарсить весь текст
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "summary" in obj:
                return obj
        except json.JSONDecodeError:
            pass

        return None

    def summarize(self, message: str) -> Optional[str]:
        """
        Генерирует summary для сообщения лога через Google Gemini API.

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
            logging.debug(f"[{self.name}] Провайдер недоступен")
            return None

        clean_message = message.strip().rstrip('.')

        # Формируем промпт
        prompt = f"Проанализируй это сообщение из системного лога и верни JSON с summary на русском языке:\n\n{clean_message}"

        try:
            # Генерация через API
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 256,
                    'top_p': 0.95,
                    'top_k': 40
                }
            )

            if not response or not response.text:
                logging.warning(f"[{self.name}] Пустой ответ от API")
                return None

            raw_text = response.text.strip()
            logging.debug(f"[{self.name}] API response: {raw_text[:200]}")

            # Извлекаем JSON
            obj = self._extract_json_from_text(raw_text)
            if obj and "summary" in obj:
                summary = str(obj["summary"]).strip()

                # Валидация: summary не должен быть пустым
                if not summary:
                    logging.warning(f"[{self.name}] Получен пустой summary")
                    return None

                logging.info(f"[{self.name}] Summary: {summary}")

                # Сохраняем в кэш
                self._set_cache(clean_message, summary)
                return summary
            else:
                logging.warning(f"[{self.name}] Не удалось извлечь JSON из ответа")
                return None

        except Exception as e:
            # Обработка специфичных ошибок API
            error_msg = str(e).lower()

            if "quota" in error_msg or "rate limit" in error_msg:
                logging.error(f"[{self.name}] Превышен лимит API: {e}")
            elif "api key" in error_msg or "authentication" in error_msg:
                logging.error(f"[{self.name}] Ошибка аутентификации: {e}")
                self._is_ready = False  # Помечаем как недоступный
            else:
                logging.error(f"[{self.name}] Ошибка запроса к API: {e}")

            return None
