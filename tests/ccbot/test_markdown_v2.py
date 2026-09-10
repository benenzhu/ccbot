"""Tests for Markdown → Telegram MarkdownV2 conversion."""

import pytest

from ccbot.markdown_v2 import _escape_mdv2, convert_markdown
from ccbot.transcript_parser import TranscriptParser

EXP_START = TranscriptParser.EXPANDABLE_QUOTE_START
EXP_END = TranscriptParser.EXPANDABLE_QUOTE_END


class TestEscapeMdv2:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "_*[]()~>#+\\-=|{}.!",
                "\\_\\*\\[\\]\\(\\)\\~\\>\\#\\+\\\\\\-\\=\\|\\{\\}\\.\\!",
            ),
            ("hello world 123", "hello world 123"),
            ("", ""),
        ],
        ids=["special-chars", "alphanumeric-unchanged", "empty-string"],
    )
    def test_escape(self, input_text: str, expected: str) -> None:
        assert _escape_mdv2(input_text) == expected


class TestConvertMarkdown:
    def test_plain_text(self) -> None:
        result = convert_markdown("hello world")
        assert "hello world" in result

    def test_bold(self) -> None:
        result = convert_markdown("**bold text**")
        assert "*bold text*" in result
        assert "**bold text**" not in result

    def test_code_block_preserved(self) -> None:
        result = convert_markdown("```python\nprint('hi')\n```")
        assert "```" in result
        assert "print" in result

    def test_expandable_quote_sentinels(self) -> None:
        text = f"{EXP_START}quoted content{EXP_END}"
        result = convert_markdown(text)
        assert EXP_START not in result
        assert EXP_END not in result
        assert ">quoted content||" in result

    def test_mixed_text_and_expandable_quote(self) -> None:
        text = f"before {EXP_START}inside quote{EXP_END} after"
        result = convert_markdown(text)
        assert EXP_START not in result
        assert EXP_END not in result
        assert ">inside quote||" in result
        assert "before" in result
        assert "after" in result


class TestTableToMarkdown:
    """table_to_markdown serializes (headers, rows) back to pipe syntax."""

    def test_basic_table(self):
        from ccbot.markdown_v2 import table_to_markdown

        md = table_to_markdown((["Name", "Value"], [["a", "1"], ["b", "2"]]))
        assert md.split("\n") == [
            "| Name | Value |",
            "|---|---|",
            "| a | 1 |",
            "| b | 2 |",
        ]

    def test_escapes_pipes_and_pads_ragged_rows(self):
        from ccbot.markdown_v2 import table_to_markdown

        md = table_to_markdown((["h1", "h2", "h3"], [["x|y"], ["1", "2", "3", "4"]]))
        lines = md.split("\n")
        assert lines[0] == "| h1 | h2 | h3 |  |"
        assert lines[1] == "|---|---|---|---|"
        assert lines[2] == "| x\\|y |  |  |  |"
        assert lines[3] == "| 1 | 2 | 3 | 4 |"

    def test_roundtrip_through_extractor(self):
        from ccbot.markdown_v2 import extract_markdown_tables, table_to_markdown

        src = "intro\n\n| A | B |\n|---|---|\n| **1** | `x` |\n\nend"
        stripped, tables = extract_markdown_tables(src)
        assert "| A |" not in stripped
        assert len(tables) == 1
        _, again = extract_markdown_tables(table_to_markdown(tables[0]))
        assert again == tables


class TestSplitTextByTables:
    """split_text_by_tables keeps tables at their original positions."""

    def test_no_tables_returns_whole_text(self):
        from ccbot.markdown_v2 import split_text_by_tables

        assert split_text_by_tables("just prose\n\nmore") == ["just prose\n\nmore"]

    def test_prose_table_prose_order(self):
        from ccbot.markdown_v2 import split_text_by_tables

        src = (
            "before\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nafter\n\n"
            "| C | D |\n|---|---|\n| 3 | 4 |"
        )
        segs = split_text_by_tables(src)
        assert segs[0] == "before\n"
        assert segs[1] == (["A", "B"], [["1", "2"]])
        assert segs[2] == "\nafter\n"
        assert segs[3] == (["C", "D"], [["3", "4"]])
        assert len(segs) == 4

    def test_drops_empty_prose_and_keeps_code_block_tables(self):
        from ccbot.markdown_v2 import split_text_by_tables

        src = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n```\n| not | a table |\n|---|---|\n```"
        )
        segs = split_text_by_tables(src)
        assert segs[0] == (["A", "B"], [["1", "2"]])
        assert isinstance(segs[1], str) and "| not | a table |" in segs[1]
        assert len(segs) == 2
