# llm/orchestrator.py
# -*- coding: utf-8 -*-
"""
Оркестратор для управления несколькими LLM-провайдерами.
Реализует цепочку вызовов с fallback между провайдерами.
"""

import logging
import re
from typing import List, Optional

from .base_provider import BaseLLMProvider


class LLMOrchestrator:
    """
    Оркестратор для управления цепочкой LLM-провайдеров.
    Пробует получить summary от каждого провайдера по очереди.
    Если все провайдеры не справились, использует rule-based fallback.
    """

    def __init__(self, providers: Optional[List[BaseLLMProvider]] = None):
        """
        Инициализирует оркестратор с цепочкой провайдеров.

        Args:
            providers: Список провайдеров в порядке приоритета
        """
        self.providers = providers or []
        self._log_provider_status()

    def _log_provider_status(self) -> None:
        """Логирует статус доступных провайдеров."""
        if not self.providers:
            logging.warning("[LLMOrchestrator] Нет доступных провайдеров!")
            return

        available = [p for p in self.providers if p.is_available]
        unavailable = [p for p in self.providers if not p.is_available]

        logging.info(f"[LLMOrchestrator] Доступно провайдеров: {len(available)}/{len(self.providers)}")
        for p in available:
            logging.info(f"  ✓ {p.name} - готов к работе")
        for p in unavailable:
            logging.warning(f"  ✗ {p.name} - недоступен")

    def add_provider(self, provider: BaseLLMProvider) -> None:
        """
        Добавляет провайдер в конец цепочки.

        Args:
            provider: Провайдер для добавления
        """
        self.providers.append(provider)
        status = "доступен" if provider.is_available else "недоступен"
        logging.info(f"[LLMOrchestrator] Добавлен провайдер {provider.name} ({status})")

    def get_summary(self, message: str) -> str:
        """
        Получает summary от первого успешного провайдера.

        Пробует провайдеры по очереди. Если все провалились,
        использует rule-based fallback.

        Args:
            message: Текст сообщения из лога

        Returns:
            str: Summary на русском языке (всегда возвращает строку)
        """
        if not message or not message.strip():
            return "Пустое сообщение лога"

        # Пробуем каждый провайдер по очереди
        for provider in self.providers:
            if not provider.is_available:
                logging.debug(f"[LLMOrchestrator] Пропуск {provider.name} (недоступен)")
                continue

            try:
                logging.debug(f"[LLMOrchestrator] Попытка получить summary от {provider.name}")
                summary = provider.summarize(message)

                if summary and summary.strip():
                    logging.info(f"[LLMOrchestrator] ✓ Summary от {provider.name}: {summary}")
                    return summary
                else:
                    logging.debug(f"[LLMOrchestrator] {provider.name} вернул пустой summary")

            except Exception as e:
                logging.error(f"[LLMOrchestrator] Ошибка в {provider.name}: {e}")
                continue

        # Все провайдеры провалились - используем fallback
        logging.warning(f"[LLMOrchestrator] Все провайдеры не справились. Используем fallback для: {message[:100]}")
        return self._rule_based_fallback(message)

    @staticmethod
    def _rule_based_fallback(message: str) -> str:
        """
        Rule-based fallback для генерации summary.
        Используется когда все LLM-провайдеры недоступны или не справились.

        Args:
            message: Текст сообщения из лога

        Returns:
            str: Простое summary на основе правил
        """
        message_lower = message.lower()

        # Паттерны ошибок
        if any(word in message_lower for word in ["error", "ошибка", "fail", "failed", "ошиб"]):
            return f"Ошибка в логе: {message[:50]}"

        # Критические сбои
        if any(word in message_lower for word in ["panic", "паника", "crash", "segfault", "fatal"]):
            return "Паника системы: критический сбой"

        # Сетевые проблемы
        if any(word in message_lower for word in ["connection", "соединение", "refused", "timeout", "unreachable"]):
            return "Проблема с сетевым соединением"

        # Проблемы с диском
        if "i/o" in message_lower or any(word in message_lower for word in ["disk", "диск", "read", "write"]):
            return "Ошибка ввода-вывода на диске"

        # Проблемы с памятью
        if any(word in message_lower for word in ["memory", "памят", "oom", "out of memory"]):
            return "Проблема с памятью системы"

        # Проблемы с процессами
        if any(word in message_lower for word in ["killed", "terminated", "signal"]):
            return "Процесс был завершен системой"

        # Проблемы с разрешениями
        if any(word in message_lower for word in ["permission", "denied", "разрешен", "доступ"]):
            return "Ошибка доступа: недостаточно прав"

        # Общий fallback
        return f"Событие в логе: {message[:50]}"

    def get_stats(self) -> dict:
        """
        Возвращает статистику по всем провайдерам.

        Returns:
            dict: Статистика работы провайдеров
        """
        stats = {
            "total_providers": len(self.providers),
            "available_providers": len([p for p in self.providers if p.is_available]),
            "providers": []
        }

        for provider in self.providers:
            provider_stats = {
                "name": provider.name,
                "available": provider.is_available,
                "cache": provider.get_cache_info()
            }
            stats["providers"].append(provider_stats)

        return stats

    def log_stats(self) -> None:
        """Выводит статистику в лог."""
        stats = self.get_stats()
        logging.info(f"[LLMOrchestrator] Статистика:")
        logging.info(f"  Всего провайдеров: {stats['total_providers']}")
        logging.info(f"  Доступно: {stats['available_providers']}")

        for pstat in stats["providers"]:
            cache_info = pstat["cache"]
            status = "✓" if pstat["available"] else "✗"
            logging.info(f"  {status} {pstat['name']}")
            if cache_info:
                logging.info(f"     Кэш: hits={cache_info.get('hits', 0)}, "
                           f"misses={cache_info.get('misses', 0)}, "
                           f"size={cache_info.get('size', 0)}")
