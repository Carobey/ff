"""Unit tests for the weekly digest builder."""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from family_finance.agents.digest import _week_window, build_digest
from family_finance.domain import Category

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


@pytest.mark.unit
def test_week_window_from_sunday_evening() -> None:
    """A scheduler run on Sunday 19:00 must summarise Mon..Sun that just ended."""
    now = datetime(2026, 5, 31, 19, 0, tzinfo=_MOSCOW)  # Sunday
    start, end = _week_window(now)
    # End is the next Monday 00:00
    assert end == datetime(2026, 6, 1, 0, 0, tzinfo=_MOSCOW)
    assert start == datetime(2026, 5, 25, 0, 0, tzinfo=_MOSCOW)
    assert (end - start).days == 7


@pytest.mark.unit
def test_week_window_from_midweek() -> None:
    """Midweek call should still target the most recently completed week."""
    now = datetime(2026, 5, 27, 14, 30, tzinfo=_MOSCOW)  # Wednesday
    start, end = _week_window(now)
    # End = next Monday 00:00
    assert end == datetime(2026, 6, 1, 0, 0, tzinfo=_MOSCOW)
    assert (end - start).days == 7


@pytest.mark.unit
def test_week_window_from_monday_targets_just_ended_week() -> None:
    """A Monday run must summarise the week that just ended, not jump forward (QA-12)."""
    now = datetime(2026, 6, 15, 9, 0, tzinfo=_MOSCOW)  # Monday
    start, end = _week_window(now)
    assert end == datetime(2026, 6, 15, 0, 0, tzinfo=_MOSCOW)
    assert start == datetime(2026, 6, 8, 0, 0, tzinfo=_MOSCOW)
    assert (end - start).days == 7


@pytest.mark.unit
async def test_build_digest_returns_none_when_no_spending() -> None:
    family_id = uuid.uuid4()
    with patch(
        "family_finance.agents.digest.PostgresTransactionRepository",
    ) as mock_cls:
        repo = mock_cls.return_value
        repo.category_breakdown = AsyncMock(return_value=[])
        text = await build_digest(family_id)
    assert text is None


@pytest.mark.unit
async def test_build_digest_composes_text_with_breakdown() -> None:
    family_id = uuid.uuid4()
    breakdown = [
        (Category.FOOD_GROCERIES, Decimal("12000"), 8),
        (Category.TRANSPORT_TAXI, Decimal("4500"), 6),
    ]

    with (
        patch("family_finance.agents.digest.PostgresTransactionRepository") as mock_cls,
        patch("family_finance.agents.digest.detect_alerts", new=AsyncMock(return_value=[])),
        patch("family_finance.agents.digest.get_chat_model") as mock_llm,
    ):
        repo = mock_cls.return_value
        repo.category_breakdown = AsyncMock(return_value=breakdown)
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="За прошедшую неделю потратили 16 500 ₽…")
        )
        text = await build_digest(family_id)

    assert text is not None
    assert "Итоги недели" in text
    assert "16 500 ₽" in text


@pytest.mark.unit
async def test_build_digest_appends_subscription_alerts() -> None:
    family_id = uuid.uuid4()
    breakdown = [(Category.FOOD_GROCERIES, Decimal("5000"), 3)]
    alerts = ["⚠️ Подписка <b>Netflix</b> подорожала: 999 ₽ вместо обычных 799 ₽ (+25%)"]

    with (
        patch("family_finance.agents.digest.PostgresTransactionRepository") as mock_cls,
        patch(
            "family_finance.agents.digest.detect_alerts",
            new=AsyncMock(return_value=alerts),
        ),
        patch("family_finance.agents.digest.get_chat_model") as mock_llm,
    ):
        repo = mock_cls.return_value
        repo.category_breakdown = AsyncMock(return_value=breakdown)
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Тратили скромно.")
        )
        text = await build_digest(family_id)

    assert text is not None
    assert "Netflix" in text
    assert "+25%" in text


@pytest.mark.unit
async def test_build_digest_falls_back_when_llm_fails() -> None:
    family_id = uuid.uuid4()
    breakdown = [(Category.FOOD_GROCERIES, Decimal("3000"), 2)]

    with (
        patch("family_finance.agents.digest.PostgresTransactionRepository") as mock_cls,
        patch("family_finance.agents.digest.detect_alerts", new=AsyncMock(return_value=[])),
        patch("family_finance.agents.digest.get_chat_model") as mock_llm,
    ):
        repo = mock_cls.return_value
        repo.category_breakdown = AsyncMock(return_value=breakdown)
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        text = await build_digest(family_id)

    assert text is not None
    # Fallback narrative still mentions amount + top category
    assert "3 000 ₽" in text
    assert Category.FOOD_GROCERIES.value in text
