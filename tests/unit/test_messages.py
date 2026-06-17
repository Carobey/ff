"""Unit tests for agents/_messages.py content helpers."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from family_finance.agents._messages import message_text, recent_dialog


@pytest.mark.unit
def test_message_text_passthrough_str() -> None:
    """A plain string content is returned unchanged."""
    assert message_text(HumanMessage(content="привет")) == "привет"


@pytest.mark.unit
def test_message_text_concatenates_text_blocks() -> None:
    """List content: text blocks and bare strings join; non-text blocks drop."""
    content = [
        {"type": "text", "text": "сумма "},
        {"type": "image_url", "image_url": {"url": "x"}},
        "за май ",
        {"type": "text", "text": "= 100"},
    ]
    assert message_text(AIMessage(content=content)) == "сумма за май = 100"


@pytest.mark.unit
def test_message_text_ignores_non_string_text_field() -> None:
    """A text block whose ``text`` is not a str is skipped, not coerced."""
    content = [{"type": "text", "text": 123}, {"type": "text", "text": "ok"}]
    assert message_text(AIMessage(content=content)) == "ok"


@pytest.mark.unit
def test_recent_dialog_filters_to_human_ai() -> None:
    """System/Tool messages are dropped; only Human/AI turns remain in order."""
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="q1"),
        ToolMessage(content="tool", tool_call_id="c1"),
        AIMessage(content="a1"),
    ]
    out = recent_dialog(messages)
    assert [m.content for m in out] == ["q1", "a1"]


@pytest.mark.unit
def test_recent_dialog_keeps_only_tail() -> None:
    """Only the last ``limit`` Human/AI turns are kept (newest at the end)."""
    messages = [HumanMessage(content=f"m{i}") for i in range(10)]
    out = recent_dialog(messages, limit=3)
    assert [m.content for m in out] == ["m7", "m8", "m9"]
