"""Tests for ccbot.transcript_parser — pure logic, no I/O."""

import pytest

from ccbot.transcript_parser import (
    ParsedMessage,
    TranscriptParser,
)

EXPQUOTE_START = TranscriptParser.EXPANDABLE_QUOTE_START
EXPQUOTE_END = TranscriptParser.EXPANDABLE_QUOTE_END
# A tool call's quote opens with blank lines that fill its collapsed preview
TOOL_START = EXPQUOTE_START + "\u2800\n" * 3


# ── parse_line ───────────────────────────────────────────────────────────


class TestParseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ('{"type": "user"}', {"type": "user"}),
            ("not-json", None),
            ("", None),
            ("   \t  ", None),
        ],
        ids=["valid_json", "invalid_json", "empty", "whitespace"],
    )
    def test_parse_line(self, line: str, expected: dict | None):
        assert TranscriptParser.parse_line(line) == expected


# ── extract_text_only ────────────────────────────────────────────────────


class TestExtractTextOnly:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("plain string", "plain string"),
            (
                [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
                "hello\nworld",
            ),
            (
                [
                    {"type": "text", "text": "keep"},
                    {"type": "tool_use", "name": "Read"},
                ],
                "keep",
            ),
            ([], ""),
            (42, ""),
        ],
        ids=["string", "text_blocks", "mixed", "empty_list", "non_list_non_string"],
    )
    def test_extract_text_only(self, content: list | str | int, expected: str):
        assert TranscriptParser.extract_text_only(content) == expected


# ── format_tool_use_summary ──────────────────────────────────────────────


class TestFormatToolUseSummary:
    @pytest.mark.parametrize(
        "name, input_data, expected",
        [
            ("Read", {"file_path": "src/main.py"}, "**Read**(src/main.py)"),
            ("Write", {"file_path": "out.txt"}, "**Write**(out.txt)"),
            ("Bash", {"command": "ls -la"}, "**Bash**(ls -la)"),
            ("Grep", {"pattern": "TODO"}, "**Grep**(TODO)"),
            ("Glob", {"pattern": "*.py"}, "**Glob**(*.py)"),
            ("Task", {"description": "analyze code"}, "**Task**(analyze code)"),
            (
                "WebFetch",
                {"url": "https://example.com"},
                "**WebFetch**(https://example.com)",
            ),
            ("WebSearch", {"query": "python async"}, "**WebSearch**(python async)"),
            ("TodoWrite", {"todos": [1, 2, 3]}, "**TodoWrite**(3 item(s))"),
            ("TodoRead", {}, "**TodoRead**"),
            (
                "AskUserQuestion",
                {"questions": [{"question": "Continue?"}]},
                "**AskUserQuestion**(Continue?)",
            ),
            ("ExitPlanMode", {}, "**ExitPlanMode**"),
            ("Skill", {"skill": "code-review"}, "**Skill**(code-review)"),
            (
                "CustomTool",
                {"first_key": "value1"},
                "**CustomTool**(value1)",
            ),
        ],
        ids=[
            "Read",
            "Write",
            "Bash",
            "Grep",
            "Glob",
            "Task",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "TodoRead",
            "AskUserQuestion",
            "ExitPlanMode",
            "Skill",
            "unknown_tool",
        ],
    )
    def test_tool_summary(self, name: str, input_data: dict, expected: str):
        assert TranscriptParser.format_tool_use_summary(name, input_data) == expected

    def test_non_dict_input(self):
        assert (
            TranscriptParser.format_tool_use_summary("Read", "not a dict") == "**Read**"
        )

    def test_summary_is_one_short_line(self):
        command = "cd /repo &&\n  " + "x" * 60
        result = TranscriptParser.format_tool_use_summary("Bash", {"command": command})
        one_line = ("cd /repo && " + "x" * 60)[:40]
        assert result == f"**Bash**({one_line}…)"
        assert TranscriptParser.tool_full_input("Bash", {"command": command}) == command
        assert TranscriptParser.tool_full_input("Bash", {"command": "ls"}) == ""


# ── extract_tool_result_text ─────────────────────────────────────────────


class TestExtractToolResultText:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("raw string", "raw string"),
            (
                [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                "line1\nline2",
            ),
            (
                [{"type": "text", "text": "keep"}, {"type": "image", "data": "..."}],
                "keep",
            ),
            (None, ""),
        ],
        ids=["string", "text_blocks", "mixed", "none"],
    )
    def test_extract_tool_result_text(self, content: str | list | None, expected: str):
        assert TranscriptParser.extract_tool_result_text(content) == expected


# ── parse_message ────────────────────────────────────────────────────────


class TestParseMessage:
    def test_user_text(self):
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="user", text="hello")

    def test_assistant_text(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi there"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="hi there")

    def test_local_command_with_stdout(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<command-name>/help</command-name>"
                            "<local-command-stdout>Available commands</local-command-stdout>"
                        ),
                    }
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command"
        assert result.text == "Available commands"
        assert result.tool_name == "/help"

    def test_local_command_invoke(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "<command-name>/clear</command-name>"}
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command_invoke"
        assert result.text == ""
        assert result.tool_name == "/clear"

    def test_non_user_assistant_returns_none(self):
        data = {
            "type": "summary",
            "message": {"content": "summary text"},
        }
        assert TranscriptParser.parse_message(data) is None

    def test_string_content(self):
        data = {
            "type": "assistant",
            "message": {"content": "plain response"},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="plain response")


# ── _format_edit_diff ────────────────────────────────────────────────────


class TestFormatEditDiff:
    @pytest.mark.parametrize(
        "old, new, check",
        [
            (
                "hello",
                "world",
                lambda r: "-hello" in r and "+world" in r,
            ),
            (
                "line1\nline2\nline3",
                "line1\nchanged\nline3",
                lambda r: "-line2" in r and "+changed" in r,
            ),
            (
                "same",
                "same",
                lambda r: r == "",
            ),
        ],
        ids=["single_line", "multi_line", "identical"],
    )
    def test_format_edit_diff(self, old: str, new: str, check):
        result = TranscriptParser._format_edit_diff(old, new)
        assert check(result), f"Check failed for ({old!r}, {new!r}): {result!r}"


# ── _format_tool_result_text ─────────────────────────────────────────────


class TestFormatToolResultText:
    @pytest.mark.parametrize(
        "text, tool_name, check",
        [
            (
                "line1\nline2\nline3",
                "Read",
                lambda r: r == "  ⎿  Read 3 lines",
            ),
            (
                "line1\nline2",
                "Write",
                lambda r: r == "  ⎿  Wrote 2 lines",
            ),
            (
                "output line",
                "Bash",
                lambda r: (
                    r.startswith("  ⎿  Output 1 lines")
                    and EXPQUOTE_START in r
                    and EXPQUOTE_END in r
                ),
            ),
            (
                "file1.py\nfile2.py\n",
                "Grep",
                lambda r: "Found 2 matches" in r and EXPQUOTE_START in r,
            ),
            (
                "a.py\nb.py\nc.py",
                "Glob",
                lambda r: "Found 3 files" in r and EXPQUOTE_START in r,
            ),
            (
                "agent says hello",
                "Task",
                lambda r: "Agent output 1 lines" in r and EXPQUOTE_START in r,
            ),
            (
                "page content here",
                "WebFetch",
                lambda r: (
                    f"Fetched {len('page content here')} characters" in r
                    and EXPQUOTE_START in r
                ),
            ),
            (
                "",
                "Read",
                lambda r: r == "",
            ),
        ],
        ids=["Read", "Write", "Bash", "Grep", "Glob", "Task", "WebFetch", "empty"],
    )
    def test_format_tool_result_text(self, text: str, tool_name: str, check):
        result = TranscriptParser._format_tool_result_text(text, tool_name)
        assert check(result), f"Failed check for {tool_name!r}: {result!r}"


# ── parse_entries ────────────────────────────────────────────────────────


class TestParseEntries:
    def test_source_index_points_at_the_entry_that_produced_each_message(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            {"type": "summary", "summary": "not a message"},
            make_jsonl_entry(
                "assistant",
                [make_text_block("Reading"), make_tool_use_block("t1", "Read")],
            ),
            make_jsonl_entry("user", [make_tool_result_block("t1", "contents")]),
            make_jsonl_entry("assistant", [make_text_block("Done")]),
        ]
        result, _ = TranscriptParser.parse_entries(entries, pending_tools={})

        assert [(e.content_type, e.source_index) for e in result] == [
            ("text", 1),
            ("tool_use", 1),
            ("tool_result", 2),
            ("text", 3),
        ]

    def test_assistant_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("assistant", [make_text_block("Hello!")])]
        result, _pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].text == "Hello!"
        assert result[0].content_type == "text"

    def test_user_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("user", [make_text_block("Hi bot")])]
        result, _pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].text == "Hi bot"

    def test_tool_use_and_result_pairing(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "app.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "file contents line1\nline2\nline3")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_use_entries = [e for e in result if e.content_type == "tool_use"]
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_use_entries) == 1
        assert tool_use_entries[0].tool_use_id == "t1"
        # The whole call is one collapsed quote: blank lines, then its summary
        assert tool_use_entries[0].text == f"{TOOL_START}Read(app.py){EXPQUOTE_END}"
        assert len(tool_result_entries) == 1
        assert tool_result_entries[0].tool_use_id == "t1"
        assert tool_result_entries[0].text == (
            f"{TOOL_START}Read(app.py) -- Read 3 lines{EXPQUOTE_END}"
        )
        assert not pending

    def test_thinking_block(self, make_jsonl_entry, make_thinking_block):
        entries = [
            make_jsonl_entry("assistant", [make_thinking_block("reasoning here")])
        ]
        result, _pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "thinking"
        assert EXPQUOTE_START in result[0].text
        assert EXPQUOTE_END in result[0].text
        assert "reasoning here" in result[0].text

    def test_local_command_with_stdout(self, make_jsonl_entry, make_text_block):
        xml = (
            "<command-name>/status</command-name>"
            "<local-command-stdout>all good</local-command-stdout>"
        )
        entries = [make_jsonl_entry("user", [make_text_block(xml)])]
        result, _pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert "/status" in result[0].text
        assert "all good" in result[0].text

    def test_exit_plan_mode_emits_plan(self, make_jsonl_entry, make_tool_use_block):
        block = make_tool_use_block(
            "t1", "ExitPlanMode", {"plan": "Step 1: do X\nStep 2: do Y"}
        )
        entries = [make_jsonl_entry("assistant", [block])]
        result, _pending = TranscriptParser.parse_entries(entries)
        texts = [e for e in result if e.content_type == "text"]
        tool_uses = [e for e in result if e.content_type == "tool_use"]
        assert len(texts) == 1
        assert "Step 1: do X" in texts[0].text
        assert len(tool_uses) >= 1

    def test_narration_is_text_and_reasoning_stays_thinking(self, make_jsonl_entry):
        """Narration: a thinking block with text, its signature tagged so."""
        import base64

        def sig(tag: bytes) -> str:
            return base64.b64encode(b"\x08\x04\x12\x05" + tag + b"\x1a\x0c").decode()

        entries = [
            make_jsonl_entry(
                "assistant",
                [
                    {"type": "thinking", "thinking": "", "signature": sig(b"x")},
                    {
                        "type": "thinking",
                        "thinking": "job 13102 已在 node 06 启动。\n\n",
                        "signature": sig(b"narration"),
                    },
                    {"type": "thinking", "thinking": "weighing options"},
                ],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)

        assert [(e.content_type, e.text) for e in result] == [
            ("narration", "job 13102 已在 node 06 启动。"),
            ("thinking", f"{EXPQUOTE_START}weighing options{EXPQUOTE_END}"),
        ]

    def test_long_command_goes_in_full_after_the_output(
        self, make_jsonl_entry, make_tool_use_block, make_tool_result_block
    ):
        command = "for n in 03 05 06 07; do ssh do-mi355x-$n nvidia-smi; done"
        entries = [
            make_jsonl_entry(
                "assistant", [make_tool_use_block("t1", "Bash", {"command": command})]
            ),
            make_jsonl_entry("user", [make_tool_result_block("t1", "GPU 0: idle")]),
        ]
        tool_use, tool_result = TranscriptParser.parse_entries(entries)[0]

        summary = f"Bash({command[:40]}…)"
        assert tool_use.text == f"{TOOL_START}{summary}\n{command}{EXPQUOTE_END}"
        assert tool_result.text == (
            f"{TOOL_START}{summary} -- Output 1 lines\n"
            f"GPU 0: idle\n\n{command}{EXPQUOTE_END}"  # output, then the command
        )

    def test_notices_claude_code_writes_as_user_entries_are_folded(
        self, make_jsonl_entry
    ):
        task = make_jsonl_entry(
            "user",
            "<task-notification>\n<task-id>ad70a95b</task-id>\n"
            "<status>completed</status>\n"
            '<summary>Agent "Watch job 13195" finished</summary>\n'
            "<result>server stuck at profile run</result>\n</task-notification>",
        )
        task["origin"] = {"kind": "task-notification"}
        peer = make_jsonl_entry(
            "user",
            "Another Claude session sent a message:\n"
            '<cross-session-message from="uds:/x.sock" from-name="EP parallel">\n'
            "Need one TP4 server tonight.\n</cross-session-message>",
        )
        peer["origin"] = {"kind": "peer"}
        human = make_jsonl_entry("user", "我们目前在做啥呢？")
        human["origin"] = {"kind": "human"}

        result, _ = TranscriptParser.parse_entries([task, peer, human])

        assert [(e.role, e.content_type) for e in result] == [
            ("assistant", "notification"),
            ("assistant", "notification"),
            ("user", "text"),
        ]
        assert result[0].text == (
            '🔔 Agent "Watch job 13195" finished\n'
            f"{EXPQUOTE_START}status: completed\n"
            f"result: server stuck at profile run{EXPQUOTE_END}"
        )
        assert result[1].text == (
            f"📨 Message from EP parallel\n"
            f"{EXPQUOTE_START}Need one TP4 server tonight.{EXPQUOTE_END}"
        )
        assert result[2].text == "我们目前在做啥呢？"

    def test_message_sent_mid_turn_is_a_user_message(self, make_jsonl_entry):
        """Claude Code queues it as an attachment, never as a user entry."""

        def queued(prompt: str, kind: str, mode: str = "prompt") -> dict:
            return {
                "type": "attachment",
                "timestamp": "2026-10-03T01:28:07.540Z",
                "attachment": {
                    "type": "queued_command",
                    "prompt": prompt,
                    "commandMode": mode,
                    "origin": {"kind": kind},
                },
            }

        entries = [
            make_jsonl_entry("assistant", [{"type": "text", "text": "Working"}]),
            queued("跑e2e好像有点慢？", "human"),
            queued(
                "<task-notification>done</task-notification>",
                "task-notification",
                "task-notification",
            ),
            queued("from another session", "peer"),
        ]
        result, _ = TranscriptParser.parse_entries(entries)

        assert [(e.role, e.text, e.source_index) for e in result] == [
            ("assistant", "Working", 0),
            ("user", "跑e2e好像有点慢？", 1),
        ]

    @staticmethod
    def _bash_that_edits(
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
        *,
        stdout: str,
        is_error: bool = False,
    ):
        """A Bash call whose result carries Claude Code's bashEditDiff."""
        result = make_jsonl_entry(
            "user",
            [make_tool_result_block("t1", stdout, is_error=is_error)],
            cwd="/repo",
        )
        result["toolUseResult"] = {
            "stdout": stdout,
            "bashEditDiff": {
                "files": [
                    {
                        "filePath": "/repo/src/q.py",
                        "hunks": [
                            {
                                "oldStart": 54,
                                "oldLines": 3,
                                "newStart": 54,
                                "newLines": 2,
                                "lines": [
                                    " keep",
                                    "-import gone",
                                    "+import new",
                                    "-import also_gone",
                                ],
                            }
                        ],
                    }
                ],
                "moreFiles": 0,
            },
        }
        return [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Bash", {"command": "sed -i x src/q.py"})],
            ),
            result,
        ]

    def test_bash_that_changed_files_shows_their_diff_before_the_output(
        self, make_jsonl_entry, make_tool_use_block, make_tool_result_block
    ):
        entries = self._bash_that_edits(
            make_jsonl_entry, make_tool_use_block, make_tool_result_block, stdout="done"
        )
        result, _ = TranscriptParser.parse_entries(entries)
        text = next(e.text for e in result if e.content_type == "tool_result")

        assert text.startswith(
            f"{TOOL_START}Bash(sed -i x src/q.py) -- Output 1 lines"
            " -- Updated src/q.py (+1 -2)\n"
        )
        body = text.removeprefix(TOOL_START).split("\n", 1)[1]
        assert body.startswith("@@ -54,3 +54,2 @@\n keep\n-import gone\n")
        assert body.index("+import new") < body.index("done")

    def test_failed_bash_still_shows_the_files_it_changed(
        self, make_jsonl_entry, make_tool_use_block, make_tool_result_block
    ):
        entries = self._bash_that_edits(
            make_jsonl_entry,
            make_tool_use_block,
            make_tool_result_block,
            stdout="Exit code 1",
            is_error=True,
        )
        result, _ = TranscriptParser.parse_entries(entries)
        text = next(e.text for e in result if e.content_type == "tool_result")

        assert " -- Error: Exit code 1 -- Updated src/q.py (+1 -2)" in text
        assert "-import gone" in text

    def test_edit_tool_diff_stats(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        edit_input = {
            "file_path": "main.py",
            "old_string": "old line",
            "new_string": "new line",
        }
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Edit", edit_input)],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "OK")],
            ),
        ]
        result, _pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        tr = tool_result_entries[0]
        assert "Added" in tr.text
        assert "removed" in tr.text
        assert EXPQUOTE_START in tr.text

    def test_error_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Bash", {"command": "rm -rf /"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "Permission denied", is_error=True)],
            ),
        ]
        result, _pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "Error: Permission denied" in tool_result_entries[0].text

    def test_interrupted_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "x.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", TranscriptParser._INTERRUPTED_TEXT)],
            ),
        ]
        result, _pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "Interrupted" in tool_result_entries[0].text

    def test_pending_tools_carry_over(self, make_jsonl_entry, make_tool_use_block):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools={})
        assert "t1" in pending
        flushed = [
            e for e in result if e.content_type == "tool_use" and e.tool_use_id == "t1"
        ]
        assert len(flushed) == 1

    def test_pending_tools_flushed_without_carry_over(
        self, make_jsonl_entry, make_tool_use_block
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, _pending = TranscriptParser.parse_entries(entries, pending_tools=None)
        tool_entries = [e for e in result if e.tool_use_id == "t1"]
        assert len(tool_entries) == 2
        assert tool_entries[0].content_type == "tool_use"
        assert tool_entries[1].content_type == "tool_use"

    def test_system_tag_filtered(self, make_jsonl_entry, make_text_block):
        entries = [
            make_jsonl_entry(
                "user",
                [
                    make_text_block(
                        "<system-reminder>secret instructions</system-reminder>"
                    )
                ],
            ),
        ]
        result, _pending = TranscriptParser.parse_entries(entries)
        user_entries = [e for e in result if e.role == "user"]
        assert len(user_entries) == 0
