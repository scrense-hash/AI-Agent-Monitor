# AI Agent Monitor - Copilot Instructions

## Architecture Overview

This is a **rsyslog AI monitoring daemon** that receives UDP syslog messages, analyzes them with LLM providers, and aggregates incidents in SQLite. The system uses a **plugin-based parser architecture** and a **multi-provider LLM orchestrator with automatic fallback**.

### Core Components

1. **`agent_daemon.py`** - Main orchestrator: UDP server → plugin dispatch → LLM analysis → SQLite storage
2. **`llm/` module** - Provider-based LLM architecture with fallback chain:
   - `base_provider.py` - Abstract base class defining the provider contract
   - `local_provider.py` - Local llama.cpp models (priority 1, with caching & retry)
   - `google_provider.py` - Google Gemini API (priority 2, with caching)
   - `orchestrator.py` - Manages provider chain, automatic fallback to rule-based summaries
3. **`plugins/`** - Syslog parsers (priority-based dispatch):
   - `xorg.py` (PRIORITY=100) - Specialized Xorg error parser
   - `syslog_default.py` (PRIORITY=1000) - Fallback for standard rsyslog format
4. **`web_monitor.py`** - Flask dashboard with auto-refresh, timezone-aware display

### Data Flow

```
UDP:1514 → agent_daemon → plugin.parse() → llm_orchestrator.get_summary() → SQLite upsert (by summary_key)
                                                    ↓
                                    LocalProvider → GoogleProvider → rule_based_fallback
```

## Critical Patterns

### 1. Plugin System (Priority-Based Dispatch)

Plugins are loaded from `plugins/*.py` and sorted by `PRIORITY` (lower = higher priority). Each must implement:

```python
PRIORITY = 100  # Lower runs first
def can_handle(line: str) -> bool: ...
def parse(line: str) -> Optional[Dict]:  # Returns: {original_line, severity, count, message, host}
```

**When adding plugins**: Set PRIORITY < 1000 to run before `syslog_default.py` fallback.

### 2. LLM Provider Contract

All providers inherit from `BaseLLMProvider` and must implement:

```python
def summarize(self, message: str) -> Optional[str]:  # Returns Russian summary or None
@property
def is_available(self) -> bool:  # Check if provider is ready
@property
def name(self) -> str:  # For logging
```

**Key behaviors**:
- Providers maintain their own caches (`_summary_cache`, `_cache_hits`, `_cache_misses`)
- `LocalProvider` retries English responses to force Russian output
- `GoogleProvider` handles API rate limits gracefully
- Orchestrator tries providers in list order, falls back to rule-based keywords

### 3. Incident Aggregation (Summary Keys)

Incidents are deduplicated by `summary_key = make_summary_key(summary, host)`:
- Same summary + same host → increment `count`, update `last_seen`
- Different host → separate incident
- Uses `ON CONFLICT(summary_key) DO UPDATE` for atomic upserts

**When modifying**: Changing `make_summary_key()` logic will affect aggregation behavior.

### 4. Severity Filtering

Only logs with `severity <= max_severity` (default 5) are processed:
- 0-2: Critical (red in UI)
- 3: Error (orange)
- 4: Warning (yellow)
- 5+: Informational (green, default cutoff)

Severity extracted from syslog PRI: `severity = PRI % 8`

## Development Workflows

### Running the System

```bash
# Activate venv (if using local model)
source venv_activate.sh

# Start daemon (auto-launches web_monitor.py)
python agent_daemon.py --model-path models/phi-3-mini.gguf --enable-google

# Test with sample log
echo "<3>$(date '+%b %d %H:%M:%S') testhost kernel: panic: test" | nc -u -w1 localhost 1514

# View logs
tail -f debug.log

# Access dashboard
open http://localhost:8000
```

### Adding a New LLM Provider

1. Create `llm/new_provider.py` inheriting `BaseLLMProvider`
2. Implement `summarize()`, `is_available`, `name` properties
3. Add caching: `_summary_cache`, `get_cache_info()`
4. Register in `agent_daemon.py`:
   ```python
   from llm import NewProvider
   providers = [LocalProvider(...), GoogleProvider(...), NewProvider(...)]
   orchestrator = LLMOrchestrator(providers)
   ```

### Adding a New Plugin

1. Create `plugins/my_plugin.py` with `PRIORITY`, `can_handle()`, `parse()`
2. Return dict with required keys: `{original_line, severity, count, message, host}`
3. Lower PRIORITY runs first (e.g., 50 for specialized, 1000 for fallback)
4. Plugin auto-loads on daemon restart (no registration needed)

### Testing LLM Providers

```python
# Test local provider
from llm import LocalProvider
provider = LocalProvider(model_path="models/phi-3-mini.gguf")
print(provider.summarize("kernel panic at boot"))  # Should return Russian summary

# Test orchestrator fallback
from llm import LLMOrchestrator
orchestrator = LLMOrchestrator([])  # No providers
print(orchestrator.get_summary("connection refused"))  # Uses rule-based fallback
```

## Configuration

### Environment Variables (Priority: CLI args > env vars > defaults)

```bash
# Database & Logging
DB_PATH=analytics.db
DEBUG_LOG_PATH=debug.log
MAX_SEVERITY_TO_PROCESS=5

# Local Model
MODEL_PATH=models/phi-3-mini.gguf
N_CTX=4096
N_THREADS=-1  # Auto-detect
N_GPU_LAYERS=0

# Google API
ENABLE_GOOGLE=true
GOOGLE_API_KEY=your-key
GOOGLE_MODEL=gemini-1.5-flash

# Database Cleanup
CLEAN_INTERVAL=604800  # 7 days in seconds
CLEAN_ON_START=false  # Set true to wipe DB on startup

# Web Monitor
TZ=Europe/Moscow  # Timezone for display
REFRESH_SEC=5
```

### Key Files

- `analytics.db` - SQLite database (WAL mode, auto-created)
- `debug.log` - Rotating log (1MB max, 3 backups)
- `models/*.gguf` - Local GGUF models for llama.cpp

## Common Pitfalls

1. **Missing `__init__.py` in `llm/`**: Module won't import. Ensure it exports providers:
   ```python
   from .local_provider import LocalProvider
   from .google_provider import GoogleProvider
   from .orchestrator import LLMOrchestrator
   ```

2. **Plugin missing required keys**: `parse()` must return all 5 keys (`original_line`, `severity`, `count`, `message`, `host`) or agent logs warning and skips.

3. **Timezone issues in web_monitor**: Uses `zoneinfo.ZoneInfo` (Python 3.9+). For older Python, install `backports.zoneinfo`. Windows may need `tzdata`.

4. **Cache invalidation**: `LocalProvider` purges cache on retry. If summary quality degrades, check `_purge_cache_key()` calls.

5. **Severity confusion**: Syslog PRI encodes both facility and severity. Always use `PRI % 8` to extract severity (0-7).

## Russian Language Requirement

All LLM summaries MUST be in Russian. `LocalProvider` detects English responses and retries with explicit translation prompt. If adding providers, implement similar language validation or rely on orchestrator's rule-based fallback (which is Russian-only).

## Migration Notes

See `MIGRATION_GUIDE.md` for details on the refactor from monolithic `agent_daemon.py` to modular `llm/` architecture. Key changes:
- LLM logic moved from 800-line daemon to separate providers
- Orchestrator handles fallback (no manual try/catch chains)
- Plugins unchanged (backward compatible)
