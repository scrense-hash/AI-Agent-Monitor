# AI-Agent Monitor

## Интеграция с Zabbix

Агент может отправлять информацию об инцидентах в Zabbix через `zabbix_sender`. Для этого задайте параметры через переменные окружения или аргументы командной строки:

```bash
python agent_daemon.py \
  --enable-zabbix-sender \
  --zabbix-server zabbix.example.com \
  --zabbix-host monitor-agent \
  --zabbix-key agent.event.summary \
  --zabbix-sender-path /usr/bin/zabbix_sender
```

Доступны переменные окружения `ENABLE_ZABBIX_SENDER`, `ZABBIX_SERVER`, `ZABBIX_HOST`, `ZABBIX_KEY`, `ZABBIX_SENDER_PATH`. Если путь до бинарника не указан, агент попытается найти его в `PATH`.

В случае ошибок отправки событие фиксируется в журнале, основной цикл обработки продолжает работу.
