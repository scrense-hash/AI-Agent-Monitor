# llm/cloud_provider.py
# -*- coding: utf-8 -*-
"""
Обобщенный облачный провайдер, совместимый с OpenAI API.
По умолчанию настроен на работу с OpenRouter (https://openrouter.ai/).
"""

import json
import logging
import os
import re
import time
from typing import Optional, Dict, Any, List

import requests
from requests import Response

from .base_provider import BaseLLMProvider

# Значения по умолчанию для облачного API
DEFAULT_CLOUD_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CLOUD_MODEL = "openai/gpt-4o-mini"


class CloudAPIError(Exception):
    """Базовое исключение для ошибок облачного API."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class CloudAPIRateLimitError(CloudAPIError):
    """Исключение для ошибок превышения лимита."""


class CloudProvider(BaseLLMProvider):
    """
    Провайдер, совместимый с OpenAI API (chat/completions).
    Может работать с OpenRouter и другими сервисами, поддерживающими API OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        request_timeout: float = 40.0,
    ):
        """
        Args:
            api_key: API ключ (если None, берется из CLOUD_API_KEY или OPENAI_API_KEY)
            model_name: Имя модели (если None, CLOUD_MODEL или значение по умолчанию)
            base_url: Базовый URL API (если None, CLOUD_API_BASE_URL или значение по умолчанию)
            proxy_host: Хост HTTP прокси
            proxy_port: Порт HTTP прокси
            request_timeout: Таймаут HTTP запросов (сек)
        """
        self._is_ready = False
        self.model_name = model_name or os.getenv("CLOUD_MODEL", DEFAULT_CLOUD_MODEL)
        self.base_url = (base_url or os.getenv("CLOUD_API_BASE_URL", DEFAULT_CLOUD_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.request_timeout = request_timeout
        self._rate_limit_attempts = 0

        # Управляемый кэш
        self._summary_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if not self.api_key:
            logging.warning("[CloudProvider] API ключ не задан. Провайдер недоступен.")
            return

        self.session = requests.Session()
        if self.proxy_host and self.proxy_port:
            proxy_url = f"http://{self.proxy_host}:{self.proxy_port}"
            self.session.proxies.update({
                "http": proxy_url,
                "https": proxy_url,
            })
            logging.info(f"[CloudProvider] Настроен HTTP прокси: {proxy_url}")

        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        self._is_ready = True
        logging.info(f"[CloudProvider] Инициализирован. Модель: {self.model_name}, базовый URL: {self.base_url}")

    @property
    def name(self) -> str:
        return f"CloudLLM({self.model_name})"

    @property
    def is_available(self) -> bool:
        return self._is_ready

    def get_cache_info(self) -> dict:
        """Возвращает статистику кэша."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._summary_cache),
        }

    def _get_from_cache(self, message: str) -> Optional[str]:
        key = message.strip().rstrip(".")
        if key in self._summary_cache:
            self._cache_hits += 1
            logging.debug(f"[{self.name}] Кэш hit: {key[:50]}")
            return self._summary_cache[key]
        self._cache_misses += 1
        return None

    def _set_cache(self, message: str, summary: str) -> None:
        key = message.strip().rstrip(".")
        self._summary_cache[key] = summary

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict]:
        """
        Извлекает JSON из текста ответа.
        """
        t = text.strip()

        # Удаляем markdown обёртки
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r'```?\s*$', '', t, flags=re.IGNORECASE | re.MULTILINE)

        json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', t, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict) and "summary" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "summary" in obj:
                return obj
        except json.JSONDecodeError:
            pass

        return None

    def summarize(self, message: str) -> Optional[str]:
        """
        Генерирует summary для сообщения лога через облачный API.
        """
        cached = self._get_from_cache(message)
        if cached is not None:
            return cached

        if not self.is_available:
            logging.debug(f"[{self.name}] Провайдер недоступен")
            return None

        clean_message = message.strip().rstrip(".")
        prompt = (
            "Проанализируй это сообщение из системного лога и верни JSON с summary на русском языке:\n\n"
            f"{clean_message}"
        )

        try:
            raw_text = self._request_completion(prompt)

            if not raw_text:
                self._rate_limit_attempts = 0
                logging.warning(f"[{self.name}] Пустой ответ от API")
                return None

            logging.debug(f"[{self.name}] API response: {raw_text[:200]}")

            obj = self._extract_json_from_text(raw_text)
            if obj and "summary" in obj:
                summary = str(obj["summary"]).strip()
                if not summary:
                    logging.warning(f"[{self.name}] Получен пустой summary")
                    return None

                logging.info(f"[{self.name}] Summary: {summary}")
                self._set_cache(clean_message, summary)
                self._rate_limit_attempts = 0
                return summary

            self._rate_limit_attempts = 0
            logging.warning(f"[{self.name}] Не удалось извлечь JSON из ответа")
            logging.warning(f"[{self.name}] Ответ API: {raw_text}")
            return None

        except CloudAPIRateLimitError as e:
            wait_seconds = self._register_rate_limit_wait(e)
            time.sleep(wait_seconds)
        except CloudAPIError as e:
            logging.error(f"[{self.name}] Ошибка запроса к API: {e}")
            self._rate_limit_attempts = 0
        except Exception as e:
            if self._is_rate_limit_error(e):
                wait_seconds = self._register_rate_limit_wait(e)
                time.sleep(wait_seconds)
            elif self._is_auth_error(e):
                logging.error(f"[{self.name}] Ошибка аутентификации: {e}")
                self._is_ready = False
            else:
                logging.error(f"[{self.name}] Необработанная ошибка API: {e}")
                self._rate_limit_attempts = 0

        return None

    def _request_completion(self, prompt: str) -> str:
        """
        Выполняет HTTP запрос к облачному API и возвращает текст ответа.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        }

        response = self._post(url, payload)
        data = self._parse_json(response)

        if "error" in data:
            error = data["error"]
            message = error.get("message", "неизвестная ошибка")
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                raise CloudAPIRateLimitError(message, status_code=status_code, payload=data)
            raise CloudAPIError(message, status_code=status_code, payload=data)

        choices = data.get("choices", [])
        if not choices:
            raise CloudAPIError("Ответ API не содержит choices", status_code=getattr(response, "status_code", None), payload=data)

        first_choice = choices[0]
        message = first_choice.get("message", {})
        raw_content = message.get("content")

        if not raw_content:
            logging.warning(f"[{self.name}] Ответ без контента: {json.dumps(data, ensure_ascii=False)[:500]}")
            return ""

        answer = self._normalize_content(raw_content, data)
        if answer:
            return answer

        # Попытка достать текст из reasoning или custom-полей, используемых некоторыми провайдерами
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            logging.warning(f"[{self.name}] Используем reasoning вместо контента")
            return reasoning.strip()

        content_list = message.get("content")
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text and str(text).strip():
                        logging.warning(f"[{self.name}] Используем content[].text вместо основного контента")
                        return str(text).strip()

        logging.warning(f"[{self.name}] Ответ без текста: {json.dumps(data, ensure_ascii=False)[:500]}")
        return ""

    def _normalize_content(self, content: Any, full_payload: Dict[str, Any]) -> str:
        """
        Унифицирует формат контента ответа (строка или список частей).
        """
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            normalized = "".join(parts).strip()
            if normalized:
                return normalized

        logging.warning(f"[{self.name}] Неожиданный формат ответа: {content!r}")
        logging.warning(f"[{self.name}] Полный ответ: {json.dumps(full_payload, ensure_ascii=False)[:500]}")
        return ""

    def _post(self, url: str, payload: Dict[str, Any]) -> Response:
        """
        Выполняет POST запрос с учетом таймаута и возможных ошибок.
        """
        try:
            response = self.session.post(url, json=payload, timeout=self.request_timeout)
        except requests.exceptions.RequestException as exc:
            raise CloudAPIError(f"Ошибка сетевого запроса: {exc}") from exc

        if response.status_code == 429:
            raise CloudAPIRateLimitError("Превышен лимит запросов", status_code=response.status_code, payload=response.text)

        if response.status_code >= 400:
            try:
                err = response.json()
            except ValueError:
                err = response.text
            raise CloudAPIError(
                f"HTTP {response.status_code}: {err}",
                status_code=response.status_code,
                payload=err,
            )

        return response

    def _parse_json(self, response: Response) -> Dict[str, Any]:
        """
        Разбирает JSON ответ API. При ошибке выбрасывает CloudAPIError.
        """
        try:
            return response.json()
        except ValueError as exc:
            raise CloudAPIError(f"Не удалось разобрать JSON: {exc}", status_code=response.status_code, payload=response.text) from exc

    def _register_rate_limit_wait(self, exc: Exception) -> int:
        """
        Увеличивает счетчик превышения лимита и рассчитывает время ожидания.
        """
        self._rate_limit_attempts += 1
        wait_minutes = self._rate_limit_attempts
        wait_seconds = wait_minutes * 60
        logging.error(
            f"[{self.name}] Превышен лимит API (попытка {self._rate_limit_attempts}): {exc}. "
            f"Повтор через {wait_minutes} мин."
        )
        return wait_seconds

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        for attr in ("status_code", "status", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int) and value == 429:
                return True
            if isinstance(value, str) and value.strip() == "429":
                return True

        response = getattr(exc, "response", None)
        if response is not None:
            try:
                status = getattr(response, "status_code", None)
                if status == 429:
                    return True
            except AttributeError:
                pass

        message = str(exc).lower()
        if "429" in message:
            return True
        if "quota" in message or "rate limit" in message or "resource exhausted" in message:
            return True

        return False

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(word in message for word in ("unauthorized", "invalid api key", "authentication", "forbidden", "401", "403"))


# Промпт для модели
SYSTEM_INSTRUCTION = """
Ты — Анализатор сообщений системных логов.

Твоя задача — понять суть сообщения из лога и вернуть ТОЛЬКО валидный JSON-объект с кратким описанием проблемы НА РУССКОМ ЯЗЫКЕ.

Формат ответа (строго JSON):
{"summary": "Полное описание на русском (максимум 30 слов)"}

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
