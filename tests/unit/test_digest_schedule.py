"""Unit tests for DigestSchedule + NLP parser."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from family_finance.agents.digest_schedule_parser import _ParsedSchedule, parse_digest_schedule
from family_finance.domain import DigestSchedule


@pytest.mark.unit
def test_to_cron_renders_5_field_expression() -> None:
    sched = DigestSchedule(day_of_week="sun", hour=19, minute=0)
    assert sched.to_cron() == "0 19 * * sun"


@pytest.mark.unit
def test_to_cron_uses_minute() -> None:
    sched = DigestSchedule(day_of_week="fri", hour=20, minute=30)
    assert sched.to_cron() == "30 20 * * fri"


@pytest.mark.unit
def test_from_cron_roundtrip() -> None:
    original = DigestSchedule(day_of_week="wed", hour=9, minute=15)
    assert DigestSchedule.from_cron(original.to_cron()) == original


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        "",
        "every monday",
        "0 19 * *",  # 4 fields, not 5
        "0 25 * * mon",  # invalid hour caught by Pydantic
    ],
)
def test_from_cron_rejects_invalid(bad: str) -> None:
    with pytest.raises((ValueError, Exception)):
        DigestSchedule.from_cron(bad)


@pytest.mark.unit
def test_human_label_russian() -> None:
    sched = DigestSchedule(day_of_week="sun", hour=19, minute=0)
    assert sched.human_label() == "по воскресеньям в 19:00"


@pytest.mark.unit
async def test_parse_digest_schedule_extracts_day_and_time() -> None:
    parsed = _ParsedSchedule(day_of_week="sun", hour=19, minute=0)
    with patch(
        "family_finance.agents.digest_schedule_parser.get_chat_model",
    ) as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=parsed
        )
        result = await parse_digest_schedule("по воскресеньям в 19:00")
    assert result == DigestSchedule(day_of_week="sun", hour=19, minute=0)


@pytest.mark.unit
async def test_parse_digest_schedule_returns_none_for_ambiguous() -> None:
    """Missing day → None → caller asks to clarify."""
    parsed = _ParsedSchedule(day_of_week=None, hour=9, minute=0)
    with patch(
        "family_finance.agents.digest_schedule_parser.get_chat_model",
    ) as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=parsed
        )
        result = await parse_digest_schedule("каждый день в 9")
    assert result is None


@pytest.mark.unit
async def test_parse_digest_schedule_returns_none_for_missing_hour() -> None:
    parsed = _ParsedSchedule(day_of_week="mon", hour=None, minute=None)
    with patch(
        "family_finance.agents.digest_schedule_parser.get_chat_model",
    ) as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=parsed
        )
        result = await parse_digest_schedule("в понедельник")
    assert result is None


@pytest.mark.unit
async def test_parse_digest_schedule_handles_llm_error() -> None:
    with patch(
        "family_finance.agents.digest_schedule_parser.get_chat_model",
    ) as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        result = await parse_digest_schedule("по воскресеньям в 19:00")
    assert result is None
