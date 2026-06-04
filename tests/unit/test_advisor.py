"""Unit tests for the AdvisorAgent: routing, 50/30/20 buckets, formatting."""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from family_finance.agents.advisor import (
    SpendingHealth,
    analyze_spending,
    bucket_of,
    build_advice_block,
    goal_status_text,
    is_advice_question,
)
from family_finance.application.ports import LedgerSummary
from family_finance.domain import Category, SavingsGoal

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")
_NOW = datetime(2026, 5, 15, 12, 0, tzinfo=_MOSCOW)


# ── is_advice_question ────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "на чём сэкономить?",
        "как мне копить?",
        "дай совет по бюджету",
        "сколько откладывать на цель?",
        "куда уходят деньги",
    ],
)
def test_is_advice_question_matches(text: str) -> None:
    assert is_advice_question(text) is True


@pytest.mark.unit
@pytest.mark.parametrize("text", ["сколько на еду в мае?", "привет", "мои подписки"])
def test_is_advice_question_skips_unrelated(text: str) -> None:
    assert is_advice_question(text) is False


# ── bucket_of ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("category", list(Category))
def test_bucket_of_is_total(category: Category) -> None:
    assert bucket_of(category) in {"needs", "wants", "other"}


@pytest.mark.unit
def test_bucket_of_specifics() -> None:
    assert bucket_of(Category.FOOD_GROCERIES) == "needs"
    assert bucket_of(Category.HOME_UTILITIES) == "needs"
    assert bucket_of(Category.FOOD_DELIVERY) == "wants"
    assert bucket_of(Category.TRANSPORT_TAXI) == "wants"
    assert bucket_of(Category.INCOME_SALARY) == "other"
    assert bucket_of(Category.UNCLASSIFIED) == "other"


# ── SpendingHealth ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_spending_health_percentages() -> None:
    h = SpendingHealth(
        income=Decimal("100000"),
        needs=Decimal("50000"),
        wants=Decimal("30000"),
        total_expenses=Decimal("80000"),
    )
    assert h.has_income is True
    assert h.needs_pct == 50
    assert h.wants_pct == 30
    assert h.savings == Decimal("20000")
    assert h.savings_pct == 20


@pytest.mark.unit
def test_spending_health_no_income_returns_none_pcts() -> None:
    h = SpendingHealth(
        income=Decimal("0"),
        needs=Decimal("10000"),
        wants=Decimal("5000"),
        total_expenses=Decimal("15000"),
    )
    assert h.has_income is False
    assert h.needs_pct is None
    assert h.savings_pct is None


# ── analyze_spending ──────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_analyze_spending_buckets_by_category() -> None:
    repo = AsyncMock()
    repo.category_breakdown = AsyncMock(
        return_value=[
            (Category.FOOD_GROCERIES, Decimal("40000"), 10),
            (Category.FOOD_DELIVERY, Decimal("15000"), 5),
            (Category.UNCLASSIFIED, Decimal("5000"), 1),
        ]
    )
    repo.aggregate = AsyncMock(return_value=LedgerSummary(total=Decimal("120000"), count=2))
    health = await analyze_spending(uuid.uuid4(), repo=repo, now=_NOW)
    assert health.needs == Decimal("40000")
    assert health.wants == Decimal("15000")
    assert health.total_expenses == Decimal("60000")
    assert health.income == Decimal("120000")
    assert health.savings == Decimal("60000")


# ── goal_status_text ──────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_goal_status_text_no_goal_hints_user() -> None:
    with patch("family_finance.agents.advisor.MCPLedgerReader") as mock_cls:
        mock_cls.return_value.get_savings_goal = AsyncMock(return_value=None)
        text = await goal_status_text(uuid.uuid4())
    assert "не задана" in text.lower()
    assert "/goal" in text


@pytest.mark.unit
async def test_goal_status_text_shows_progress() -> None:
    fam = uuid.uuid4()
    goal = SavingsGoal(
        family_id=fam,
        target_amount=Decimal("200000"),
        target_date=date(2026, 12, 31),
        created_at=datetime(2026, 1, 1, tzinfo=_MOSCOW),
    )
    with patch("family_finance.agents.advisor.MCPLedgerReader") as mock_cls:
        inst = mock_cls.return_value
        inst.get_savings_goal = AsyncMock(return_value=goal)
        inst.net_cashflow = AsyncMock(return_value=Decimal("50000"))
        text = await goal_status_text(fam, now=_NOW)
    assert "200 000" in text
    assert "25%" in text  # 50000 / 200000


# ── build_advice_block (digest) ───────────────────────────────────────────────


@pytest.mark.unit
async def test_build_advice_block_none_when_no_data() -> None:
    with patch("family_finance.agents.advisor.MCPLedgerReader") as mock_cls:
        inst = mock_cls.return_value
        inst.category_breakdown = AsyncMock(return_value=[])
        inst.aggregate = AsyncMock(return_value=LedgerSummary(total=Decimal("0"), count=0))
        inst.get_savings_goal = AsyncMock(return_value=None)
        block = await build_advice_block(uuid.uuid4(), now=_NOW)
    assert block is None


@pytest.mark.unit
async def test_build_advice_block_flags_low_savings() -> None:
    with patch("family_finance.agents.advisor.MCPLedgerReader") as mock_cls:
        inst = mock_cls.return_value
        inst.category_breakdown = AsyncMock(
            return_value=[
                (Category.FOOD_DELIVERY, Decimal("30000"), 8),
                (Category.FOOD_GROCERIES, Decimal("40000"), 10),
            ]
        )
        inst.aggregate = AsyncMock(return_value=LedgerSummary(total=Decimal("80000"), count=2))
        inst.get_savings_goal = AsyncMock(return_value=None)
        block = await build_advice_block(uuid.uuid4(), now=_NOW)
    assert block is not None
    # savings = 80000 - 70000 = 10000 → 12% < 20% norm; biggest want = delivery
    assert "Наставник" in block
    assert "food.delivery" in block
