"""Unit tests for the BudgetsAgent: parsing, formatting, alerts."""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from family_finance.agents.budgets import (
    current_moscow_month,
    detect_budget_alerts,
    format_budgets,
    is_budgets_question,
    parse_budget_category,
)
from family_finance.domain import Budget, BudgetStatus, Category

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


def _status(
    *,
    category: Category = Category.FOOD_GROCERIES,
    limit: str = "30000",
    spent: str = "0",
) -> BudgetStatus:
    return BudgetStatus(
        budget=Budget(
            family_id=uuid.uuid4(),
            category=category,
            monthly_limit=Decimal(limit),
        ),
        spent_this_month=Decimal(spent),
    )


# ── current_moscow_month ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_current_moscow_month_mid_month() -> None:
    now = datetime(2026, 5, 15, 14, 0, tzinfo=_MOSCOW)
    start, end = current_moscow_month(now)
    assert start == datetime(2026, 5, 1, 0, 0, tzinfo=_MOSCOW)
    assert end == datetime(2026, 6, 1, 0, 0, tzinfo=_MOSCOW)


@pytest.mark.unit
def test_current_moscow_month_december_rolls_year() -> None:
    now = datetime(2026, 12, 20, 0, 0, tzinfo=_MOSCOW)
    start, end = current_moscow_month(now)
    assert start == datetime(2026, 12, 1, 0, 0, tzinfo=_MOSCOW)
    assert end == datetime(2027, 1, 1, 0, 0, tzinfo=_MOSCOW)


# ── parse_budget_category ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("food.groceries", Category.FOOD_GROCERIES),
        ("продукты", Category.FOOD_GROCERIES),
        ("ЕДА", Category.FOOD_GROCERIES),
        ("ЖКХ", Category.HOME_UTILITIES),
        ("коммуналка", Category.HOME_UTILITIES),
        ("аптека", Category.HEALTH_PHARMACY),
        ("такси", Category.TRANSPORT_TAXI),
    ],
)
def test_parse_budget_category_recognises(raw: str, expected: Category) -> None:
    assert parse_budget_category(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "wat", "xyz", "12345"])
def test_parse_budget_category_returns_none_for_unknown(raw: str) -> None:
    assert parse_budget_category(raw) is None


# ── is_budgets_question ───────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["мои бюджеты", "покажи лимиты", "show budgets"],
)
def test_is_budgets_question_matches(text: str) -> None:
    assert is_budgets_question(text) is True


@pytest.mark.unit
def test_is_budgets_question_skips_unrelated() -> None:
    assert is_budgets_question("привет, что нового?") is False


# ── format_budgets ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_budgets_empty_hints_user() -> None:
    text = format_budgets([])
    assert "не настроены" in text.lower()


@pytest.mark.unit
def test_format_budgets_green_under_warn_threshold() -> None:
    text = format_budgets([_status(limit="30000", spent="10000")])  # 33%
    assert "🟢" in text
    assert "33%" in text


@pytest.mark.unit
def test_format_budgets_yellow_at_warn() -> None:
    text = format_budgets([_status(limit="30000", spent="25000")])  # 83%
    assert "🟡" in text


@pytest.mark.unit
def test_format_budgets_red_over_limit() -> None:
    text = format_budgets([_status(limit="30000", spent="33000")])  # 110%
    assert "🔴" in text


# ── BudgetStatus.pct / over_budget properties ─────────────────────────────────


@pytest.mark.unit
def test_budget_status_pct_and_over_budget() -> None:
    s = _status(limit="20000", spent="22000")
    assert s.pct == 110
    assert s.over_budget is True


@pytest.mark.unit
def test_budget_status_not_over_at_exactly_warn_threshold() -> None:
    s = _status(limit="10000", spent="8000")
    assert s.pct == 80
    assert s.over_budget is False


# ── detect_budget_alerts ──────────────────────────────────────────────────────


@pytest.mark.unit
async def test_detect_budget_alerts_returns_empty_when_all_green() -> None:
    statuses = [_status(limit="30000", spent="10000")]  # 33%
    with patch(
        "family_finance.agents.budgets.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.get_budget_status = AsyncMock(return_value=statuses)
        alerts = await detect_budget_alerts(uuid.uuid4())
    assert alerts == []


@pytest.mark.unit
async def test_detect_budget_alerts_flags_warn() -> None:
    statuses = [_status(limit="10000", spent="9000")]  # 90%
    with patch(
        "family_finance.agents.budgets.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.get_budget_status = AsyncMock(return_value=statuses)
        alerts = await detect_budget_alerts(uuid.uuid4())
    assert len(alerts) == 1
    assert "🟡" in alerts[0]


@pytest.mark.unit
async def test_detect_budget_alerts_flags_over() -> None:
    statuses = [_status(limit="10000", spent="12000")]  # 120%
    with patch(
        "family_finance.agents.budgets.MCPLedgerReader",
    ) as mock_cls:
        mock_cls.return_value.get_budget_status = AsyncMock(return_value=statuses)
        alerts = await detect_budget_alerts(uuid.uuid4())
    assert len(alerts) == 1
    assert "🔴" in alerts[0]
    assert "превышен" in alerts[0]
