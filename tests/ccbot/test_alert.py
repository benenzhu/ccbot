"""Tests for alert — the WeCom message sent when flood control stops ccbot."""

import json
from unittest.mock import MagicMock, patch

from ccbot import alert


def _reply(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response


def test_posts_a_wecom_text_message():
    with (
        patch.object(alert.config, "alert_webhook_url", "https://hook/send?key=k"),
        patch.object(
            alert.urllib.request, "urlopen", return_value=_reply({"errcode": 0})
        ) as urlopen,
    ):
        assert alert.send_alert("ccbot stopped: flood") is True

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://hook/send?key=k"
    assert json.loads(request.data) == {
        "msgtype": "text",
        "text": {"content": "ccbot stopped: flood"},
    }


def test_rejected_or_unreachable_webhook_reports_failure():
    with patch.object(alert.config, "alert_webhook_url", "https://hook"):
        with patch.object(
            alert.urllib.request,
            "urlopen",
            return_value=_reply({"errcode": 93000, "errmsg": "invalid webhook url"}),
        ):
            assert alert.send_alert("x") is False
        with patch.object(alert.urllib.request, "urlopen", side_effect=OSError("down")):
            assert alert.send_alert("x") is False


def test_no_webhook_configured_sends_nothing():
    with (
        patch.object(alert.config, "alert_webhook_url", ""),
        patch.object(alert.urllib.request, "urlopen") as urlopen,
    ):
        assert alert.send_alert("x") is False
    urlopen.assert_not_called()
