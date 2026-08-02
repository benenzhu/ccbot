"""Unit tests for tts — text-to-speech synthesis providers."""

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccbot import tts


@pytest.fixture(autouse=True)
def _reset_client():
    """Ensure each test starts with a fresh client."""
    tts._client = None
    yield
    tts._client = None


@pytest.fixture
def mock_config():
    """Patch config with test values."""
    with patch.object(tts, "config") as cfg:
        cfg.tts_provider = "openai"
        cfg.tts_model = "gpt-4o-mini-tts"
        cfg.tts_voice = ""
        cfg.tts_max_chars = 3000
        cfg.tts_segment_chars = 180
        cfg.tts_speed = 1.0
        cfg.openai_api_key = "sk-test-key"
        cfg.openai_base_url = "https://api.openai.com/v1"
        cfg.azure_speech_key = "azure-test-key"
        cfg.azure_speech_region = "eastasia"
        cfg.volcano_tts_app_id = "123456"
        cfg.volcano_tts_access_key = "volc-test-key"
        cfg.volcano_tts_api_key = ""
        cfg.volcano_tts_resource_id = "seed-tts-2.0"
        yield cfg


class TestPrepareTtsText:
    def test_code_blocks_dropped(self, mock_config):
        text = "before\n```python\nprint(1)\n```\nafter"
        assert tts.prepare_tts_text(text) == "before\n\nafter"

    def test_markdown_stripped(self, mock_config):
        text = "# Title\n\n**bold** and *italic* and `code` and [link](https://x.com)"
        assert (
            tts.prepare_tts_text(text) == "Title\n\nbold and italic and code and link"
        )

    def test_list_and_quote_markers_stripped(self, mock_config):
        text = "- item one\n1. item two\n> quoted"
        assert tts.prepare_tts_text(text) == "item one\nitem two\nquoted"

    def test_truncated_to_max_chars(self, mock_config):
        mock_config.tts_max_chars = 10
        assert tts.prepare_tts_text("a" * 100) == "a" * 10

    def test_all_code_yields_empty(self, mock_config):
        assert tts.prepare_tts_text("```\nonly code\n```") == ""


class TestSplitForTts:
    def test_short_text_stays_whole(self):
        assert tts.split_for_tts("短句。", 180) == ["短句。"]

    def test_empty_text_yields_nothing(self):
        assert tts.split_for_tts("", 180) == []

    def test_zero_limit_disables_splitting(self):
        text = "一" * 500
        assert tts.split_for_tts(text, 0) == [text]

    def test_splits_on_cjk_terminators(self):
        text = "第一句话在这里。第二句话也在这里！第三句话结束了？"
        segments = tts.split_for_tts(text, 14)
        assert segments == [
            "第一句话在这里。",
            "第二句话也在这里！",
            "第三句话结束了？",
        ]

    def test_segments_respect_limit(self):
        text = "这是一个句子。" * 40
        segments = tts.split_for_tts(text, 60)
        assert len(segments) > 1
        assert all(len(s) <= 60 for s in segments)

    def test_no_content_lost(self):
        text = "克隆完成。这个架构新东西很多！我先读代码，再讲结构。"
        assert "".join(tts.split_for_tts(text, 12)) == text

    def test_cjk_joined_without_space(self):
        text = "第一句。第二句。"
        assert tts.split_for_tts(text, 180) == [text]

    def test_ascii_sentences_keep_space(self):
        text = "First one. Second one."
        assert tts.split_for_tts(text, 180) == ["First one. Second one."]

    def test_abbreviations_and_decimals_not_split(self):
        text = "Version 3.5 is out, e.g. the new one. Next sentence here."
        assert tts.split_for_tts(text, 40) == [
            "Version 3.5 is out, e.g. the new one.",
            "Next sentence here.",
        ]

    def test_runaway_sentence_hard_split(self):
        text = "啊" * 500
        segments = tts.split_for_tts(text, 180)
        assert all(len(s) <= 180 for s in segments)
        assert "".join(segments) == text

    def test_hard_split_prefers_soft_punctuation(self):
        text = "前面这段话相当长，后面这段话也不短，还有第三段内容在这里"
        segments = tts.split_for_tts(text, 16)
        assert all(len(s) <= 16 for s in segments)
        assert segments[0].endswith("，")


class TestPrepareTtsSegments:
    def test_strips_then_splits(self, mock_config):
        mock_config.tts_segment_chars = 10
        text = "**第一句话在这里。**\n\n```python\ncode\n```\n\n第二句话在这里。"
        assert tts.prepare_tts_segments(text) == [
            "第一句话在这里。",
            "第二句话在这里。",
        ]

    def test_short_reply_stays_one_segment(self, mock_config):
        text = "**第一句话在这里。**\n\n第二句话在这里。"
        assert len(tts.prepare_tts_segments(text)) == 1

    def test_unspeakable_yields_empty_list(self, mock_config):
        assert tts.prepare_tts_segments("```\nonly code\n```") == []

    def test_respects_max_chars_before_splitting(self, mock_config):
        mock_config.tts_max_chars = 10
        mock_config.tts_segment_chars = 4
        segments = tts.prepare_tts_segments("啊" * 100)
        assert "".join(segments) == "啊" * 10


class TestParseVolcanoStream:
    def test_joins_audio_chunks(self):
        b64a = base64.b64encode(b"OggS-part1").decode()
        b64b = base64.b64encode(b"part2").decode()
        body = (
            f'{{"code": 0, "data": "{b64a}"}}\n'
            f'{{"code": 0, "data": "{b64b}"}}\n'
            '{"code": 20000000}'
        )
        assert tts._parse_volcano_stream(body) == b"OggS-part1part2"

    def test_error_code_raises(self):
        body = '{"code": 55000000, "message": "resource ID is mismatched"}'
        with pytest.raises(tts.TtsApiError, match="55000000"):
            tts._parse_volcano_stream(body)

    def test_blank_and_invalid_lines_skipped(self):
        b64 = base64.b64encode(b"audio").decode()
        body = f'\nnot-json\n{{"code": 0, "data": "{b64}"}}\n\n{{"code": 20000000}}'
        assert tts._parse_volcano_stream(body) == b"audio"


def _mock_response(
    *, content: bytes = b"", text: str = "", status_code: int = 200
) -> httpx.Response:
    """Build a fake httpx.Response."""
    request = httpx.Request("POST", "https://example.com/tts")
    return httpx.Response(
        status_code=status_code, content=content or text.encode(), request=request
    )


class TestSynthesizeSpeech:
    @pytest.mark.asyncio
    async def test_openai_provider(self, mock_config):
        resp = _mock_response(content=b"OggS-fake-opus")
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            audio = await tts.synthesize_speech("你好世界")

        assert audio == b"OggS-fake-opus"
        url_arg = mock_post.call_args[0][0]
        assert url_arg == "https://api.openai.com/v1/audio/speech"
        body = mock_post.call_args.kwargs["json"]
        assert body["response_format"] == "opus"
        assert body["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_azure_provider(self, mock_config):
        mock_config.tts_provider = "azure"
        resp = _mock_response(content=b"OggS-azure-opus")
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            audio = await tts.synthesize_speech("你好")

        assert audio == b"OggS-azure-opus"
        url_arg = mock_post.call_args[0][0]
        assert "eastasia.tts.speech.microsoft.com" in url_arg
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Microsoft-OutputFormat"] == "ogg-48khz-16bit-mono-opus"
        assert b"zh-CN-XiaoxiaoNeural" in mock_post.call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_volcano_provider(self, mock_config):
        mock_config.tts_provider = "volcano"
        b64 = base64.b64encode(b"OggS-volc").decode()
        resp = _mock_response(text=f'{{"code": 0, "data": "{b64}"}}')
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            audio = await tts.synthesize_speech("你好")

        assert audio == b"OggS-volc"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Api-App-Id"] == "123456"
        assert headers["X-Api-Access-Key"] == "volc-test-key"
        assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"
        req = mock_post.call_args.kwargs["json"]["req_params"]
        assert req["speaker"] == "zh_female_vv_uranus_bigtts"
        assert req["audio_params"]["format"] == "ogg_opus"

    @pytest.mark.asyncio
    async def test_volcano_speed_mapped_to_speech_rate(self, mock_config):
        mock_config.tts_provider = "volcano"
        mock_config.tts_speed = 1.3
        b64 = base64.b64encode(b"x").decode()
        resp = _mock_response(text=f'{{"code": 0, "data": "{b64}"}}')
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            await tts.synthesize_speech("你好")

        audio_params = mock_post.call_args.kwargs["json"]["req_params"]["audio_params"]
        assert audio_params["speech_rate"] == 30

    @pytest.mark.asyncio
    async def test_volcano_default_speed_omits_speech_rate(self, mock_config):
        mock_config.tts_provider = "volcano"
        b64 = base64.b64encode(b"x").decode()
        resp = _mock_response(text=f'{{"code": 0, "data": "{b64}"}}')
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            await tts.synthesize_speech("你好")

        audio_params = mock_post.call_args.kwargs["json"]["req_params"]["audio_params"]
        assert "speech_rate" not in audio_params

    @pytest.mark.asyncio
    async def test_volcano_api_key_auth(self, mock_config):
        mock_config.tts_provider = "volcano"
        mock_config.volcano_tts_app_id = ""
        mock_config.volcano_tts_api_key = "new-console-key"
        b64 = base64.b64encode(b"x").decode()
        resp = _mock_response(text=f'{{"code": 0, "data": "{b64}"}}')
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            await tts.synthesize_speech("你好")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Api-Key"] == "new-console-key"
        assert "X-Api-App-Id" not in headers

    @pytest.mark.asyncio
    async def test_unspeakable_text_raises(self, mock_config):
        with pytest.raises(ValueError, match="No speakable text"):
            await tts.synthesize_speech("```\ncode only\n```")

    @pytest.mark.asyncio
    async def test_empty_audio_raises(self, mock_config):
        resp = _mock_response(content=b"")
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(tts.TtsApiError, match="Empty audio"):
                await tts.synthesize_speech("你好")

    @pytest.mark.asyncio
    async def test_api_error_raises(self, mock_config):
        resp = _mock_response(text="unauthorized", status_code=401)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await tts.synthesize_speech("你好")


class TestCloseClient:
    @pytest.mark.asyncio
    async def test_close_client_when_open(self):
        tts._client = httpx.AsyncClient()
        await tts.close_client()
        assert tts._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        assert tts._client is None
        await tts.close_client()
        assert tts._client is None
