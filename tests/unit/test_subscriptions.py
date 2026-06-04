"""Unit tests for the SubscriptionAgent and the formatter / alert helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from family_finance.agents.subscriptions import (
    detect_alerts,
    format_subscriptions,
    is_subscriptions_question,
    subscriptions_node,
)
from family_finance.domain import Category, Subscription


def _sub(
    *,
    merchant: str = "Netflix",
    average: str = "799",
    last: str = "799",
    cadence: int = 30,
    occurrences: int = 6,
) -> Subscription:
    return Subscription(
        merchant=merchant,
        category=Category.ENTERTAINMENT_SUBS,
        cadence_days=cadence,
        average_amount=Decimal(average),
        last_amount=Decimal(last),
        last_seen=datetime(2026, 5, 1, tzinfo=UTC),
        occurrences=occurrences,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "мои подписки",
        "сколько у меня регулярных трат?",
        "повторяющиеся платежи",
        "ежемесячные списания",
        "/subscriptions",
    ],
)
def test_is_subscriptions_question_matches(text: str) -> None:
    assert is_subscriptions_question(text) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "сколько на еду в мае?",
        "когда я последний раз покупал кофе?",
        "привет",
    ],
)
def test_is_subscriptions_question_skips_unrelated(text: str) -> None:
    assert is_subscriptions_question(text) is False


@pytest.mark.unit
def test_format_subscriptions_renders_stable_price() -> None:
    text = format_subscriptions([_sub()])
    assert "Netflix" in text
    assert "799 ₽" in text
    assert "раз в ~30 дн." in text


@pytest.mark.unit
def test_format_subscriptions_shows_price_delta() -> None:
    sub = _sub(average="799", last="999")
    text = format_subscriptions([sub])
    # Should mention both numbers and a +25% delta
    assert "999 ₽" in text
    assert "обычно 799 ₽" in text
    assert "+25%" in text


@pytest.mark.unit
async def test_subscriptions_node_reports_none_when_empty() -> None:
    family_id = str(uuid.uuid4())
    state = {"family_id": family_id, "messages": [HumanMessage(content="мои подписки")]}

    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        repo = mock_cls.return_value
        repo.detect_recurring = AsyncMock(return_value=[])
        result = await subscriptions_node(state)  # type: ignore[arg-type]

    assert result["current_intent"] == "idle"
    assert "Не нашёл" in str(result["messages"][0].content)


@pytest.mark.unit
async def test_subscriptions_node_renders_list() -> None:
    family_id = str(uuid.uuid4())
    state = {"family_id": family_id, "messages": [HumanMessage(content="подписки")]}

    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        repo = mock_cls.return_value
        repo.detect_recurring = AsyncMock(return_value=[_sub()])
        result = await subscriptions_node(state)  # type: ignore[arg-type]

    content = str(result["messages"][0].content)
    assert "Netflix" in content
    assert "Нашёл регулярных трат: 1" in content


@pytest.mark.unit
async def test_detect_alerts_skips_short_baselines() -> None:
    """Subscription with <4 occurrences must NOT produce an alert."""
    sub = _sub(occurrences=3, average="799", last="1500")
    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.detect_recurring = AsyncMock(return_value=[sub])
        alerts = await detect_alerts(family_id=uuid.uuid4())
    assert alerts == []


@pytest.mark.unit
async def test_detect_alerts_skips_small_deltas() -> None:
    """Sub-15% price change is too noisy to alert on."""
    sub = _sub(occurrences=6, average="800", last="850")  # +6.25%
    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.detect_recurring = AsyncMock(return_value=[sub])
        alerts = await detect_alerts(family_id=uuid.uuid4())
    assert alerts == []


@pytest.mark.unit
async def test_detect_alerts_flags_price_hike() -> None:
    sub = _sub(occurrences=6, average="799", last="999")  # +25%
    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.detect_recurring = AsyncMock(return_value=[sub])
        alerts = await detect_alerts(family_id=uuid.uuid4())
    assert len(alerts) == 1
    assert "подорожала" in alerts[0]
    assert "Netflix" in alerts[0]


@pytest.mark.unit
async def test_detect_alerts_flags_price_drop() -> None:
    sub = _sub(occurrences=6, average="999", last="699")  # -30%
    with patch(
        "family_finance.agents.subscriptions.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.detect_recurring = AsyncMock(return_value=[sub])
        alerts = await detect_alerts(family_id=uuid.uuid4())
    assert len(alerts) == 1
    assert "подешевела" in alerts[0]
