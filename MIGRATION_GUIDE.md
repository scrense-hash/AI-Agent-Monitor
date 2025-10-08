# Руководство по миграции на модульную архитектуру LLM

## Обзор изменений

Вся логика работы с LLM вынесена из `agent_daemon.py` в отдельный модуль `llm/` с провайдерами и оркестратором.

## Шаг 1: Структура файлов

### Было:
```
rsyslog-ai-agent/
├── agent_daemon.py       # ~800 строк (включая LLM-логику)
├── web_monitor.py
├── plugins/
│   ├── syslog_default.py
│   └── xorg.py
└── models/
    └── phi-3-mini.gguf
```

### Стало:
```
rsyslog-ai-agent/
├── agent_daemon.py       # ~400 строк (только оркестрация)
├── web_monitor.py
├── llm/                  # НОВАЯ ПАПКА
│   ├── __init__.py
│   ├── base_provider.py
│   ├── local_provider.py
│   ├── google_provider.py
│   └── orchestrator.py
├── plugins/
│   ├── syslog_default.py
│   └── xorg.py
└── models/
    └── phi-3-mini.gguf
```

## Шаг 2: Создание модуля LLM

1. **Создайте папку `llm/`:**
```bash
mkdir llm
```

2. **Скопируйте файлы модуля:**
   - `llm/__init__.py`
   - `llm/base_provider.py`
   - `llm/local_provider.py`
   - `llm/google_provider.py`
   - `llm/orchestrator.py`

## Шаг 3: Замена agent_daemon.py

### Старый код (было):

```python
class LocalLLMClient:
    def __init__(self, model_path, ...):
        self.llama = Llama(...)
        # ... 300+ строк кода ...

    def summarize(self, message):
        # ... сложная логика ...
        pass

class Agent:
    def __init__(self, cfg):
        self.llm = LocalLLMClient(...)  # Прямое использование

    def serve(self):
        # ...
        summary = self.llm.summarize(message)
```

### Новый код (стало):

```python
from llm import LocalProvider, GoogleProvider, LLMOrchestrator

class Agent:
    def __init__(self, cfg):
        # Создаем провайдеры
        providers = []

        local = LocalProvider(cfg.model_path, ...)
        if local.is_available:
            providers.append(local)

        if cfg.enable_google:
            google = GoogleProvider(cfg.google_api_key)
            if google.is_available:
                providers.append(google)

        # Оркестратор управляет всем
        self.llm_orchestrator = LLMOrchestrator(providers)

    def serve(self):
        # ...
        summary = self.llm_orchestrator.get_summary(message)
```

## Шаг 4: Обновление зависимостей

### Проверьте установленные пакеты:

```bash
# Базовые зависимости (уже были)
pip install llama-cpp-python
pip install flask
pip install waitress

# Опциональные для локального провайдера
pip install json-repair      # для repair JSON
pip install langid           # для детекции языка
pip install langdetect       # альтернатива langid

# Для Google провайдера (если используете)
pip install google-generativeai
```

## Шаг 5: Конфигурация

### Старая конфигурация (работает как раньше):

```bash
python agent_daemon.py \
  --model-path models/phi-3-mini.gguf \
  --n-ctx 4096 \
  --n-threads 8
```

### Новая конфигурация (с Google API):

```bash
# Через переменные окружения
export ENABLE_GOOGLE=true
export GOOGLE_API_KEY="your-api-key-here"
export GOOGLE_MODEL="gemini-1.5-flash"

python agent_daemon.py \
  --model-path models/phi-3-mini.gguf \
  --n-ctx 4096 \
  --enable-google
```

Или через аргументы:

```bash
python agent_daemon.py \
  --model-path models/phi-3-mini.gguf \
  --enable-google \
  --google-api-key "your-key" \
  --google-model "gemini-1.5-flash"
```

## Шаг 6: Проверка работы

### 1. Запустите агента:

```bash
python agent_daemon.py
```

### Ожидаемый вывод:

```
============================================================
RSYSLOG AI-AGENT - REFACTORED VERSION
============================================================
База данных: analytics.db
Лог отладки: debug.log
Max severity: 5
...
КОНФИГУРАЦИЯ LLM:
  Локальная модель: models/phi-3-mini.gguf
  Контекст: 4096, Потоки: -1, GPU слои: 0
  Google API: ВЫКЛЮЧЕН (или ВКЛЮЧЕН)
============================================================
Инициализация LLM-провайдеров...
============================================================
[LocalProvider] Загрузка модели: models/phi-3-mini.gguf
[LocalProvider] Модель успешно загружена
[LLMOrchestrator] Доступно провайдеров: 1/1
  ✓ LocalLlama - готов к работе
============================================================
Загружен плагин: syslog_default (PRIORITY=1000)
Загружен плагин: xorg (PRIORITY=100)
Агент запущен на 0.0.0.0:1514, БД: analytics.db
```

### 2. Отправьте тестовый лог:

```bash
echo "<3>$(date '+%b %d %H:%M:%S') testhost kernel: panic: test" | \
  nc -u -w1 localhost 1514
```

### 3. Проверьте логи:

```bash
tail -f debug.log
```

Ожидаемый вывод:

```
[LLMOrchestrator] Попытка получить summary от LocalLlama
[LocalLlama] Inference time (attempt 1): 0.85s
[LocalLlama] Summary language: RU (0.92, langid) | Паника системы...
[LLMOrchestrator] ✓ Summary от LocalLlama: Паника системы: критический сбой
```

## Шаг 7: Проверка Google API (опционально)

Если включили Google провайдер:

```bash
export ENABLE_GOOGLE=true
export GOOGLE_API_KEY="your-key"

python agent_daemon.py --enable-google
```

Ожидаемый вывод:

```
[GoogleProvider] Инициализирован с моделью gemini-1.5-flash
[LLMOrchestrator] Доступно провайдеров: 2/2
  ✓ LocalLlama - готов к работе
  ✓ GoogleGemini(gemini-1.5-flash) - готов к работе
```

При обработке логов:

```
[LLMOrchestrator] Попытка получить summary от LocalLlama
[LocalLlama] Кэш hit: ...
[LLMOrchestrator] ✓ Summary от LocalLlama: ...
```

Если локальная модель не справится:

```
[LLMOrchestrator] Попытка получить summary от LocalLlama
[LocalLlama] All retries failed...
[LLMOrchestrator] Попытка получить summary от GoogleGemini
[GoogleGemini] Summary: Ошибка чтения данных
[LLMOrchestrator] ✓ Summary от GoogleGemini: ...
```

## Шаг 8: Тестирование fallback

### Тест 1: Отключите локальную модель

```bash
# Укажите несуществующий путь
python agent_daemon.py --model-path /nonexistent/model.gguf --enable-google
```

Агент должен работать только через Google API.

### Тест 2: Все провайдеры недоступны

```bash
# Без локальной модели и без Google
python agent_daemon.py --model-path /nonexistent/model.gguf
```

Агент должен использовать rule-based fallback:

```
[LLMOrchestrator] НЕТ ДОСТУПНЫХ LLM-ПРОВАЙДЕРОВ!
[LLMOrchestrator] Все провайдеры не справились. Используем fallback...
Summary: Ошибка в логе: kernel panic...
```

## Шаг 9: Мониторинг статистики

Агент автоматически выводит статистику каждые 5 минут:

```
[LLMOrchestrator] Статистика:
  Всего провайдеров: 2
  Доступно: 2
  ✓ LocalLlama
     Кэш: hits=142, misses=58, size=58
  ✓ GoogleGemini(gemini-1.5-flash)
     Кэш: hits=23, misses=5, size=5
```

При остановке (Ctrl+C):

```
============================================================
ФИНАЛЬНАЯ СТАТИСТИКА LLM:
[LLMOrchestrator] Статистика:
  Всего провайдеров: 2
  Доступно: 2
  ...
============================================================
Агент остановлен.
```

## Откат на старую версию (если нужно)

Если возникли проблемы, можно откатиться:

1. **Сохраните резервную копию:**
```bash
mv agent_daemon.py agent_daemon_new.py
git checkout agent_daemon.py  # или используйте старую версию
```

2. **Удалите модуль llm:**
```bash
rm -rf llm/
```

3. **Запустите старую версию:**
```bash
python agent_daemon.py
```

## Частые проблемы

### Проблема 1: ModuleNotFoundError: No module named 'llm'

**Решение:**
```bash
# Убедитесь, что папка llm/ в правильном месте
ls -la llm/

# Должны быть файлы:
# __init__.py
# base_provider.py
# local_provider.py
# google_provider.py
# orchestrator.py
```

### Проблема 2: ImportError: cannot import name 'LocalProvider'

**Решение:**
```bash
# Проверьте __init__.py
cat llm/__init__.py

# Должно быть:
# from .local_provider import LocalProvider
# from .google_provider import GoogleProvider
# ...
```

### Проблема 3: Google API не работает

**Решение:**
```bash
# Проверьте ключ API
echo $GOOGLE_API_KEY

# Проверьте флаг enable-google
python agent_daemon.py --enable-google

# Проверьте логи
grep "GoogleProvider" debug.log
```

### Проблема 4: Локальная модель не загружается

**Решение:**
```bash
# Проверьте путь к модели
ls -lh models/phi-3-mini.gguf

# Проверьте llama-cpp-python
python -c "from llama_cpp import Llama; print('OK')"

# Проверьте логи
grep "LocalProvider" debug.log
```

## Преимущества новой архитектуры

✅ **Чистота кода**: agent_daemon.py уменьшился с ~800 до ~400 строк
✅ **Модульность**: Каждый провайдер - отдельный файл
✅ **Расширяемость**: Добавление OpenAI = 1 новый файл
✅ **Отказоустойчивость**: Автоматический fallback между провайдерами
✅ **Тестируемость**: Можно тестировать каждый компонент отдельно
✅ **Совместимость**: Старая конфигурация работает без изменений

## Следующие шаги

1. ✅ Протестируйте работу с локальной моделью
2. ✅ Попробуйте Google API (если нужно)
3. ✅ Мониторьте статистику кэша
4. 📝 Добавьте новых провайдеров (OpenAI, Anthropic) при необходимости
5. 📝 Настройте production конфигурацию

## Получение помощи

Если возникли проблемы:

1. Проверьте логи: `tail -f debug.log`
2. Проверьте статус провайдеров в выводе при запуске
3. Используйте `--help` для списка всех опций
4. Создайте issue с логами и конфигурацией