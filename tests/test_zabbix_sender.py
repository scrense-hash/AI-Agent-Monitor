import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from integrations.zabbix_sender import DEFAULT_TIMEOUT_SECONDS, ZabbixSenderError, send_event


def _make_config(**overrides):
    defaults = dict(
        zabbix_sender_path="/usr/bin/zabbix_sender",
        zabbix_server="zabbix.example.com",
        zabbix_host="monitor-agent",
        zabbix_key="agent.event",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_send_event_invokes_subprocess_with_expected_arguments():
    config = _make_config()
    payload = {
        "summary": "Disk space low",
        "severity": 3,
        "host": "server01",
        "count": 5,
    }

    with mock.patch("integrations.zabbix_sender.subprocess.run") as mock_run, mock.patch(
        "integrations.zabbix_sender.shutil.which", return_value=config.zabbix_sender_path
    ):
        mock_run.return_value = mock.Mock(stdout="sent", stderr="")
        send_event(config=config, timeout=DEFAULT_TIMEOUT_SECONDS, **payload)

    args = mock_run.call_args[0][0]
    assert args[0] == config.zabbix_sender_path
    assert args[1:3] == ["-z", config.zabbix_server]
    assert args[3:5] == ["-s", config.zabbix_host]
    assert args[5:7] == ["-k", config.zabbix_key]
    assert args[7] == "-o"
    metric_body = args[8]
    assert "Disk space low" in metric_body
    assert "\"severity\": 3" in metric_body
    assert "\"count\": 5" in metric_body


def test_send_event_raises_when_sender_missing():
    config = _make_config(zabbix_sender_path=None)

    with mock.patch("integrations.zabbix_sender.shutil.which", return_value=None):
        with pytest.raises(ZabbixSenderError):
            send_event("summary", 2, "host", 1, config)


def test_send_event_wraps_called_process_error():
    config = _make_config()

    with mock.patch("integrations.zabbix_sender.shutil.which", return_value=config.zabbix_sender_path), mock.patch(
        "integrations.zabbix_sender.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=[config.zabbix_sender_path], stderr="failed"),
    ):
        with pytest.raises(ZabbixSenderError) as exc:
            send_event("summary", 4, "host", 1, config)
        assert "failed" in str(exc.value)
