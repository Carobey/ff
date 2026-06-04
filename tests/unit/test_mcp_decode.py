"""Unit tests for the MCP payload decoder."""

from __future__ import annotations

import pytest

from family_finance.infrastructure.mcp.client import _decode_payload


@pytest.mark.unit
def test_decode_empty_list_is_empty_result() -> None:
    # FastMCP passes an empty structured list straight through (no text block).
    # Regression: this used to raise ValueError and crash subscriptions/budgets
    # on accounts with no data.
    assert _decode_payload([]) == []


@pytest.mark.unit
def test_decode_structured_list_passthrough() -> None:
    rows = [{"merchant": "Netflix", "cadence_days": 30}]
    assert _decode_payload(rows) == rows


@pytest.mark.unit
def test_decode_text_content_block() -> None:
    raw = [{"type": "text", "text": '{"total": "0"}'}]
    assert _decode_payload(raw) == {"total": "0"}


@pytest.mark.unit
def test_decode_json_string() -> None:
    assert _decode_payload('{"a": 1}') == {"a": 1}


@pytest.mark.unit
def test_decode_dict_passthrough() -> None:
    assert _decode_payload({"a": 1}) == {"a": 1}


@pytest.mark.unit
def test_decode_unexpected_type_raises() -> None:
    with pytest.raises(ValueError, match="Unexpected MCP tool result"):
        _decode_payload(42)
