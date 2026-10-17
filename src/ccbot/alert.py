"""Out-of-band alert — a WeCom group-bot message for when Telegram can't be used.

When Telegram flood control stops ccbot (rate_limit.py), the topics can't say
so; main posts the reason to CCBOT_ALERT_WEBHOOK (a WeCom group-bot webhook URL)
on the way out. Unset means no alert.

Key function: send_alert(text) -> bool.
"""

import json
import logging
import urllib.request

from .config import config

logger = logging.getLogger(__name__)

ALERT_TIMEOUT = 10.0  # seconds


def send_alert(text: str) -> bool:
    """Post text to the alert webhook; True if WeCom accepted it."""
    url = config.alert_webhook_url
    if not url:
        return False
    body = json.dumps({"msgtype": "text", "text": {"content": text}}).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=ALERT_TIMEOUT) as response:
            reply = json.loads(response.read() or b"{}")
    except (OSError, ValueError) as e:
        logger.error("Alert not sent: %s", e)
        return False
    if reply.get("errcode") != 0:
        logger.error("Alert rejected: %s", reply)
        return False
    logger.info("Alert sent")
    return True
