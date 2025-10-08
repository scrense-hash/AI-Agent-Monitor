# llm/base_provider.py
# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для всех LLM-провайдеров.
Определяет общий контракт для получения summary из логов.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """
    Абстрактный базовый класс для всех LLM-провайдеров.
    Все провайдеры должны реализовать методы summarize и свойство is_available.
    """

    @abstractmethod
    def summarize(self, message: str) -> Optional[str]:
        """
        Принимает сообщение лога и возвращает краткое описание (summary) на русском языке.

        Args:
            message: Текст сообщения из лога для анализа

        Returns:
            Optional[str]: Summary на русском языке или None в случае неудачи
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Проверяет готовность провайдера к работе.

        Returns:
            bool: True если провайдер готов (модель загружена, API ключ есть и т.д.)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Возвращает имя провайдера для логирования.

        Returns:
            str: Имя провайдера (например, "LocalLlama", "GoogleGemini")
        """
        pass

    def get_cache_info(self) -> dict:
        """
        Возвращает информацию о кэше провайдера (если есть).

        Returns:
            dict: Статистика кэша или пустой словарь
        """
        return {}
