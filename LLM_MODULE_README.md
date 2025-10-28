# LLM Модуль - Документация

## Обзор

Рефакторинг вынес всю LLM-логику в отдельный модуль `llm/` с модульной архитектурой провайдеров. Это обеспечивает:

- ✅ **Чистоту кода**: `agent_daemon.py` теперь занимается только оркестрацией
- ✅ **Расширяемость**: Легко добавить новых провайдеров (OpenAI, Anthropic и т.д.)
- ✅ **Тестируемость**: Каждый провайдер можно тестировать независимо
- ✅ **Отказоустойчивость**: Автоматический fallback между провайдерами
- ✅ **Управление зависимостями**: Каждый провайдер изолирован

## Структура модуля

```
llm/
├── __init__.py              # Экспорты модуля
├── base_provider.py         # Базовый абстрактный класс
├── local_provider.py        # Локальная модель (llama.cpp)
├── cloud_provider.py        # Облачный провайдер (OpenAI совместимый)
└── orchestrator.py          # Оркестратор с fallback
```

## Архитектура

### 1. Базовый провайдер (`base_provider.py`)

Определяет контракт для всех провайдеров:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def summarize(self, message: str) -> Optional[str]:
        """Возвращает summary или None при неудаче"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True если провайдер готов к работе"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя провайдера для логирования"""
        pass
```

### 2. Локальный провайдер (`local_provider.py`)

**Возможности:**
- Загрузка локальной GGUF модели через llama.cpp
- Кэширование результатов
- Автоматический retry для русского языка
- Детекция языка (langid/langdetect + эвристики)
- Rule-based fallback

**Пример использования:**
```python
from llm import LocalProvider

provider = LocalProvider(
    model_path="models/phi-3-mini-4k-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=-1,  # автоопределение
    n_gpu_layers=0
)

if provider.is_available:
    summary = provider.summarize("kernel panic at boot")
    print(summary)  # "Паника ядра: критический сбой системы"
```

### 3. Облачный провайдер (`cloud_provider.py`)

**Возможности:**
- Совместим с OpenAI API (chat/completions), по умолчанию использует OpenRouter
- Настраиваемые `base_url`, модель и ключ через переменные окружения
- Поддержка HTTP-прокси и инкрементального ожидания при `429`
- Кэширование результатов и извлечение JSON даже из частичных ответов
- Расширенные логи, включая полный ответ API при ошибке парсинга

**Пример использования:**
```python
from llm import CloudProvider

provider = CloudProvider(
    api_key="your-api-key",
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4o-mini"
)

if provider.is_available:
    summary = provider.summarize("connection timeout")
    print(summary)  # "Проблема с сетевым соединением"
```

**Переменные окружения:**
- `CLOUD_API_KEY` (или `OPENAI_API_KEY`) — API ключ
- `CLOUD_API_BASE_URL=https://openrouter.ai/api/v1` — базовый URL
- `CLOUD_MODEL=openai/gpt-4o-mini` — модель по умолчанию
- `ENABLE_CLOUD=true` — включить провайдер

### 4. Оркестратор (`orchestrator.py`)

**Возможности:**
- Управление цепочкой провайдеров
- Автоматический fallback при сбоях
- Сбор статистики по всем провайдерам
- Rule-based fallback если все провайдеры недоступны

**Пример использования:**
```python
from llm import LocalProvider, CloudProvider, LLMOrchestrator

# Создаем провайдеры
local = LocalProvider(model_path="...")
cloud = CloudProvider()

# Собираем оркестратор (порядок = приоритет)
orchestrator = LLMOrchestrator([local, cloud])

# Получаем summary (пробует провайдеры по очереди)
summary = orchestrator.get_summary("disk I/O error")

# Статистика
orchestrator.log_stats()
```

## Использование в agent_daemon.py

Обновленный `agent_daemon.py` теперь намного проще:

```python
from llm import LocalProvider, CloudProvider, LLMOrchestrator

class Agent:
    def __init__(self, cfg: Config):
        # Инициализация провайдеров
        providers = []

        # Локальная модель (приоритет 1)
        local = LocalProvider(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers
        )
        if local.is_available:
            providers.append(local)

        # Облачный API (приоритет 2)
        if cfg.enable_cloud:
            cloud = CloudProvider(
                api_key=cfg.cloud_api_key,
                base_url=cfg.cloud_api_base_url,
                model_name=cfg.cloud_model
            )
            if cloud.is_available:
                providers.append(cloud)

        # Создаем оркестратор
        self.llm_orchestrator = LLMOrchestrator(providers)

    def process_log(self, message: str):
        # Просто получаем summary - вся сложность скрыта
        summary = self.llm_orchestrator.get_summary(message)
        # ... сохраняем в БД
```

## Добавление нового провайдера

Пример добавления OpenAI провайдера:

```python
# llm/openai_provider.py
from .base_provider import BaseLLMProvider
import openai

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self._is_ready = True

    @property
    def name(self) -> str:
        return "OpenAI-GPT4"

    @property
    def is_available(self) -> bool:
        return self._is_ready

    def summarize(self, message: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "Ты анализируешь логи. Верни JSON с summary на русском."
                }, {
                    "role": "user",
                    "content": message
                }]
            )
            # ... парсинг JSON ...
            return summary
        except Exception as e:
            logging.error(f"OpenAI error: {e}")
            return None
```

Затем в `agent_daemon.py`:

```python
from llm import LocalProvider, CloudProvider, OpenAIProvider, LLMOrchestrator

providers = [
    LocalProvider(...),      # Приоритет 1
    CloudProvider(...),      # Приоритет 2
    OpenAIProvider(...),     # Приоритет 3
]
orchestrator = LLMOrchestrator(providers)
```

## Конфигурация

### Через переменные окружения:

```bash
# Локальная модель
export MODEL_PATH="/path/to/model.gguf"
export N_CTX=4096
export N_THREADS=8
export N_GPU_LAYERS=35

# Облачный API
export ENABLE_CLOUD=true
export CLOUD_API_KEY="your-key"
export CLOUD_API_BASE_URL="https://openrouter.ai/api/v1"
export CLOUD_MODEL="openai/gpt-4o-mini"
```

### Через аргументы командной строки:

```bash
python agent_daemon.py \
  --model-path models/phi-3-mini.gguf \
  --n-ctx 4096 \
  --n-threads 8 \
  --enable-cloud \
  --cloud-model openai/gpt-4o-mini
```

## Логирование

Каждый провайдер логирует свою работу с префиксом:

```
[LocalLlama] Модель успешно загружена
[LocalLlama] Summary language: RU (0.92, langid) | Ошибка чтения данных
[CloudLLM(openai/gpt-4o-mini)] Summary: Проблема с памятью системы
[LLMOrchestrator] ✓ Summary от LocalLlama: Паника ядра...
```

## Статистика

Оркестратор собирает статистику:

```python
stats = orchestrator.get_stats()
# {
#   "total_providers": 2,
#   "available_providers": 2,
#   "providers": [
#     {
#       "name": "LocalLlama",
#       "available": true,
#       "cache": {"hits": 142, "misses": 58, "size": 58}
#     },
#     {
#       "name": "CloudLLM(openai/gpt-4o-mini)",
#       "available": true,
#       "cache": {"hits": 23, "misses": 5, "size": 5}
#     }
#   ]
# }
```

## Тестирование

Создайте mock провайдер для тестов:

```python
# tests/mock_provider.py
from llm import BaseLLMProvider

class MockProvider(BaseLLMProvider):
    def __init__(self, responses: dict):
        self.responses = responses
        self._is_ready = True

    @property
    def name(self) -> str:
        return "MockProvider"

    @property
    def is_available(self) -> bool:
        return self._is_ready

    def summarize(self, message: str) -> Optional[str]:
        return self.responses.get(message, "Мок summary")

# Использование в тестах
mock = MockProvider({
    "error": "Тестовая ошибка",
    "panic": "Тестовая паника"
})
orchestrator = LLMOrchestrator([mock])
assert orchestrator.get_summary("error") == "Тестовая ошибка"
```

## Зависимости

### Локальный провайдер:
```bash
pip install llama-cpp-python
pip install json-repair  # опционально
pip install langid       # опционально, для детекции языка
pip install langdetect   # опционально, альтернатива langid
```

### Облачный провайдер:
```bash
pip install requests
```

## FAQ

**Q: Что если все провайдеры недоступны?**
A: Оркестратор использует rule-based fallback на основе ключевых слов.

**Q: Можно ли использовать только один провайдер?**
A: Да, просто передайте список с одним провайдером в оркестратор.

**Q: Как изменить порядок приоритета?**
A: Измените порядок в списке providers при создании оркестратора.

**Q: Поддерживается ли кэширование?**
A: Да, каждый провайдер имеет свой собственный кэш результатов.

**Q: Как отключить локальную модель?**
A: Не добавляйте LocalProvider в список провайдеров или используйте только флаг enable-cloud.

## Миграция со старой версии

1. Создайте папку `llm/` в корне проекта
2. Скопируйте новые файлы модуля
3. Замените старый `agent_daemon.py` на рефакторенную версию
4. Установите дополнительные зависимости (если нужны)
5. Настройте переменные окружения для облачного API (если используете)

Старая конфигурация останется совместимой!

## Преимущества новой архитектуры

1. **Изоляция**: Каждый провайдер независим
2. **Расширяемость**: Добавление нового провайдера = один файл
3. **Отказоустойчивость**: Автоматический fallback
4. **Чистота**: agent_daemon.py стал в 2 раза короче
5. **Тестируемость**: Можно тестировать каждый компонент отдельно
6. **DRY**: Нет дублирования кода между провайдерами
