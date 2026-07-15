"""Text-to-speech synthesis for reading Claude's final reply aloud.

Converts the final assistant message of each turn into an OGG/Opus voice
clip (the format Telegram voice bubbles require — no ffmpeg needed).
Providers, selected via CCBOT_TTS_PROVIDER:
  - "openai": POST /audio/speech (gpt-4o-mini-tts), response_format=opus
  - "azure": Azure Speech REST API, ogg-48khz-16bit-mono-opus output
  - "volcano": 火山引擎豆包语音 v3 单向流式接口, format=ogg_opus
    (NDJSON response; base64 audio chunks are joined into one stream)

Markdown is stripped before synthesis (code blocks dropped entirely) and
input is capped at config.tts_max_chars.

Key function: synthesize_speech(text) -> bytes (OGG/Opus)
"""

import base64
import json
import logging
import re
from xml.sax.saxutils import escape as xml_escape

import httpx

from .config import config

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


class TtsApiError(Exception):
    """The TTS provider returned an error or unusable audio.

    Distinct from ValueError (nothing speakable in the text, permanent):
    these failures are typically transient and worth one retry.
    """


_OPENAI_DEFAULT_VOICE = "alloy"
_AZURE_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
# 2.0 voices carry the _uranus_bigtts suffix and pair with seed-tts-2.0;
# 1.0 voices (_moon_bigtts etc.) need VOLCANO_TTS_RESOURCE_ID=seed-tts-1.0
_VOLCANO_DEFAULT_VOICE = "zh_female_vv_uranus_bigtts"
_VOLCANO_TTS_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
# Terminal code marking the end of the volcano NDJSON stream
_VOLCANO_STREAM_END = 20000000

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|~~)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_QUOTE_PREFIX_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)


def _get_client() -> httpx.AsyncClient:
    """Return a lazily-initialized httpx client singleton."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


def prepare_tts_text(text: str) -> str:
    """Strip markdown down to speakable plain text.

    Code blocks are unspeakable and dropped entirely; inline code, links,
    headings, emphasis, quotes, and list markers keep their text content.
    The result is capped at config.tts_max_chars.
    """
    text = _CODE_BLOCK_RE.sub("", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    # Run twice so nested emphasis (***bold italic***) fully unwraps
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _QUOTE_PREFIX_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[: config.tts_max_chars]


async def _synthesize_openai(text: str) -> bytes:
    """Synthesize via OpenAI /audio/speech; opus format is OGG-contained."""
    url = f"{config.openai_base_url.rstrip('/')}/audio/speech"
    payload: dict[str, object] = {
        "model": config.tts_model,
        "voice": config.tts_voice or _OPENAI_DEFAULT_VOICE,
        "input": text,
        "response_format": "opus",
    }
    if config.tts_speed != 1.0:
        payload["speed"] = config.tts_speed
    response = await _get_client().post(
        url,
        headers={"Authorization": f"Bearer {config.openai_api_key}"},
        json=payload,
    )
    response.raise_for_status()
    return response.content


async def _synthesize_azure(text: str) -> bytes:
    """Synthesize via Azure Speech REST API (free F0 tier friendly)."""
    url = (
        f"https://{config.azure_speech_region}"
        ".tts.speech.microsoft.com/cognitiveservices/v1"
    )
    voice = config.tts_voice or _AZURE_DEFAULT_VOICE
    body = xml_escape(text)
    if config.tts_speed != 1.0:
        body = f"<prosody rate='{(config.tts_speed - 1) * 100:+.0f}%'>{body}</prosody>"
    ssml = (
        "<speak version='1.0' xml:lang='zh-CN'>"
        f"<voice name='{voice}'>{body}</voice>"
        "</speak>"
    )
    response = await _get_client().post(
        url,
        headers={
            "Ocp-Apim-Subscription-Key": config.azure_speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "ogg-48khz-16bit-mono-opus",
            "User-Agent": "ccbot",
        },
        content=ssml.encode("utf-8"),
    )
    response.raise_for_status()
    return response.content


def _parse_volcano_stream(body: str) -> bytes:
    """Join base64 audio chunks from a volcano NDJSON response body.

    Each line is a JSON object: {"code": 0, "data": "<base64>"} carries
    audio; code 20000000 ends the stream; anything else is an API error.
    Chunks are transport slices of one encoded stream, so plain
    concatenation yields a valid OGG/Opus file.
    """
    chunks: list[bytes] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = obj.get("code", -1)
        if code == 0:
            data = obj.get("data")
            if data:
                chunks.append(base64.b64decode(data))
        elif code != _VOLCANO_STREAM_END:
            raise TtsApiError(
                f"Volcano TTS error {code}: {obj.get('message', 'unknown')}"
            )
    return b"".join(chunks)


async def _synthesize_volcano(text: str) -> bytes:
    """Synthesize via 火山引擎豆包语音 v3 unidirectional streaming API."""
    headers = {"X-Api-Resource-Id": config.volcano_tts_resource_id}
    if config.volcano_tts_app_id:
        headers["X-Api-App-Id"] = config.volcano_tts_app_id
        headers["X-Api-Access-Key"] = config.volcano_tts_access_key
    else:
        headers["X-Api-Key"] = config.volcano_tts_api_key
    audio_params: dict[str, object] = {"format": "ogg_opus", "sample_rate": 24000}
    if config.tts_speed != 1.0:
        # speech_rate is [-50, 100]: 100 = 2.0x speed, -50 = 0.5x
        rate = round((config.tts_speed - 1) * 100)
        audio_params["speech_rate"] = max(-50, min(100, rate))
    response = await _get_client().post(
        _VOLCANO_TTS_URL,
        headers=headers,
        json={
            "user": {"uid": "ccbot"},
            "req_params": {
                "text": text,
                "speaker": config.tts_voice or _VOLCANO_DEFAULT_VOICE,
                "audio_params": audio_params,
            },
        },
    )
    response.raise_for_status()
    return _parse_volcano_stream(response.text)


async def synthesize_speech(text: str) -> bytes:
    """Convert text to an OGG/Opus voice clip via the configured provider.

    Raises:
        ValueError: If nothing speakable remains after markdown stripping
            (permanent — do not retry).
        TtsApiError: If the provider returns an error or empty audio.
        httpx.HTTPStatusError: On HTTP-level API errors (401, 429, 5xx).
    """
    speakable = prepare_tts_text(text)
    if not speakable:
        raise ValueError("No speakable text after stripping markdown")

    if config.tts_provider == "azure":
        audio = await _synthesize_azure(speakable)
    elif config.tts_provider == "volcano":
        audio = await _synthesize_volcano(speakable)
    else:
        audio = await _synthesize_openai(speakable)

    if not audio:
        raise TtsApiError("Empty audio returned by TTS API")
    return audio


async def close_client() -> None:
    """Close the httpx client (call on shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
