"""Text-to-speech synthesis for reading Claude's replies aloud.

Converts an assistant message into one or more OGG/Opus voice clips (the
format Telegram voice bubbles require — no ffmpeg needed).
Providers, selected via CCBOT_TTS_PROVIDER:
  - "openai": POST /audio/speech (gpt-4o-mini-tts), response_format=opus
  - "azure": Azure Speech REST API, ogg-48khz-16bit-mono-opus output
  - "volcano": 火山引擎豆包语音 v3 单向流式接口, format=ogg_opus
    (NDJSON response; base64 audio chunks are joined into one stream)

Markdown is stripped before synthesis (code blocks dropped entirely) and
input is capped at config.tts_max_chars. Long replies are split at
sentence boundaries into config.tts_segment_chars-sized segments so the
caller can synthesize and deliver them incrementally — the first voice
bubble then arrives after one short segment instead of the whole reply.

Key functions:
  - prepare_tts_segments(text) -> list[str] (speakable, split for latency)
  - synthesize_prepared(segment) -> bytes (OGG/Opus, no re-stripping)
  - synthesize_speech(text) -> bytes (strip + synthesize in one call)
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
_VOLCANO_TTS_URL = "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional"
# Terminal code marking the end of the volcano NDJSON stream
_VOLCANO_STREAM_END = 20000000

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|~~)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_QUOTE_PREFIX_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)

# Sentence-ish boundaries for segmenting long replies. CJK terminators
# stand alone (Chinese text has no space after 。), while ASCII ones need
# trailing whitespace so "e.g." and "3.5" stay intact. A newline is also a
# break, so list items and paragraphs never straddle segments.
_SENTENCE_SPLIT_RE = re.compile(
    r"""(?<=[。！？；…])              # CJK terminator — no space needed
      | (?<=[.!?;])(?=\s)             # ASCII terminator + whitespace
      | \n+                           # line / paragraph break
    """,
    re.VERBOSE,
)


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


def _hard_split(chunk: str, limit: int) -> list[str]:
    """Split an over-long sentence that has no usable boundary.

    Prefers commas and other soft punctuation, falling back to a blunt
    character slice so a single runaway sentence can never exceed limit.
    """
    pieces: list[str] = []
    rest = chunk
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(
            window.rfind("，"),
            window.rfind("、"),
            window.rfind(","),
            window.rfind("："),
            window.rfind(":"),
            window.rfind(" "),
        )
        # Ignore a boundary so early that segments would be lopsided
        if cut < limit // 2:
            cut = limit - 1
        pieces.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].lstrip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def _join_chunks(left: str, right: str) -> str:
    """Concatenate two sentences, spacing them only if either side is ASCII.

    CJK text needs no separator after 。/！/？, but English sentences do.
    """
    if not left:
        return right
    sep = "" if _is_cjk(left[-1]) and _is_cjk(right[0]) else " "
    return f"{left}{sep}{right}"


def _is_cjk(char: str) -> bool:
    """True for CJK ideographs and CJK punctuation."""
    return "　" <= char <= "鿿" or "＀" <= char <= "￯"


def split_for_tts(text: str, limit: int) -> list[str]:
    """Pack speakable text into segments of at most `limit` characters.

    Splits on sentence boundaries and greedily fills each segment, so
    segments stay whole-sentence wherever the text allows. A limit of 0
    or less disables splitting.
    """
    if limit <= 0 or len(text) <= limit:
        return [text] if text else []

    segments: list[str] = []
    current = ""
    for chunk in _SENTENCE_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) > limit:
            # Flush what we have, then break the oversized sentence apart
            if current:
                segments.append(current)
                current = ""
            segments.extend(_hard_split(chunk, limit))
            continue
        candidate = _join_chunks(current, chunk)
        if len(candidate) > limit:
            segments.append(current)
            current = chunk
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments


def prepare_tts_segments(text: str) -> list[str]:
    """Strip markdown, then split into synthesis-sized segments.

    Returns an empty list when nothing speakable remains.
    """
    speakable = prepare_tts_text(text)
    if not speakable:
        return []
    return split_for_tts(speakable, config.tts_segment_chars)


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


async def synthesize_prepared(speakable: str) -> bytes:
    """Synthesize already-stripped text via the configured provider.

    Use with prepare_tts_segments(), which does the stripping once for
    the whole reply before splitting it.

    Raises:
        ValueError: If the text is empty (permanent — do not retry).
        TtsApiError: If the provider returns an error or empty audio.
        httpx.HTTPStatusError: On HTTP-level API errors (401, 429, 5xx).
    """
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


async def synthesize_speech(text: str) -> bytes:
    """Convert text to a single OGG/Opus voice clip (strip + synthesize).

    Raises:
        ValueError: If nothing speakable remains after markdown stripping
            (permanent — do not retry).
        TtsApiError: If the provider returns an error or empty audio.
        httpx.HTTPStatusError: On HTTP-level API errors (401, 429, 5xx).
    """
    return await synthesize_prepared(prepare_tts_text(text))


async def close_client() -> None:
    """Close the httpx client (call on shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
