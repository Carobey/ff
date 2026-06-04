"""Integration tests for the family-finance MCP server.

These exercise a real round-trip: the test process spawns the FastMCP server
over stdio via ``langchain-mcp-adapters``, the tool runs against Postgres, and
the JSON payload comes back decoded. This is the same path the LangGraph
``ledger`` node uses, so it guards the MCP consumer wiring end-to-end.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import asyncpg
import pytest

from family_finance.domain import Category, Currency, Direction, Transaction, TransactionSource
from family_finance.infrastructure.mcp import MCPLedgerReader, call_finance_tool
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import get_settings


async def _skip_if_postgres_unavailable() -> None:
    try:
        conn = await asyncpg.connect(dsn=get_settings().database_url.get_secret_value())
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres is unavailable: {exc}")
    else:
        await conn.close()


@pytest.mark.integration
async def test_aggregate_spending_tool_round_trip() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"MCP {telegram_user_id}",
    )
    await repo.add_many(
        [
            Transaction(
                family_id=family_id,
                member_id=member_id,
                occurred_at=datetime(2026, 4, 10, tzinfo=UTC),
                amount=Decimal("100.00"),
                currency=Currency.RUB,
                direction=Direction.EXPENSE,
                merchant_raw="MCP pharmacy",
                category=Category.HEALTH_PHARMACY,
                confidence=1.0,
                source=TransactionSource.BANK_CSV,
                import_hash=f"mcp:{uuid.uuid4()}",
            ),
        ]
    )

    result = await call_finance_tool(
        "aggregate_spending",
        {
            "family_id": str(family_id),
            "categories": [Category.HEALTH_PHARMACY.value],
            "directions": [Direction.EXPENSE.value],
            "start": "2026-04-01T00:00:00+00:00",
            "end": "2026-05-01T00:00:00+00:00",
        },
    )

    assert Decimal(str(result["total"])) == Decimal("100.00")
    assert result["count"] == 1


@pytest.mark.integration
async def test_goal_status_tool_round_trip() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"MCP {telegram_user_id}",
    )

    no_goal = await call_finance_tool("goal_status", {"family_id": str(family_id)})
    assert no_goal["has_goal"] is False

    await repo.set_savings_goal(family_id=family_id, target_amount=Decimal("200000"))
    with_goal = await call_finance_tool("goal_status", {"family_id": str(family_id)})

    assert with_goal["has_goal"] is True
    assert Decimal(str(with_goal["target_amount"])) == Decimal("200000")


@pytest.mark.integration
async def test_reader_budget_status_round_trip() -> None:
    """MCPLedgerReader rebuilds BudgetStatus from the budget_status tool."""
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"MCP {telegram_user_id}",
    )
    await repo.set_budget(
        family_id=family_id,
        category=Category.FOOD_GROCERIES,
        monthly_limit=Decimal("30000"),
    )
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    await repo.add_many(
        [
            Transaction(
                family_id=family_id,
                member_id=member_id,
                occurred_at=month_start.replace(day=5),
                amount=Decimal("12000.00"),
                currency=Currency.RUB,
                direction=Direction.EXPENSE,
                merchant_raw="MCP grocery",
                category=Category.FOOD_GROCERIES,
                confidence=1.0,
                source=TransactionSource.BANK_CSV,
                import_hash=f"mcp-budget:{uuid.uuid4()}",
            ),
        ]
    )
    month_end = month_start.replace(
        year=month_start.year + (month_start.month // 12),
        month=(month_start.month % 12) + 1,
    )

    statuses = await MCPLedgerReader().get_budget_status(
        family_id=family_id,
        month_start=month_start,
        month_end=month_end,
    )

    grocery = next(s for s in statuses if s.budget.category is Category.FOOD_GROCERIES)
    assert grocery.budget.monthly_limit == Decimal("30000")
    assert grocery.spent_this_month == Decimal("12000.00")


@pytest.mark.integration
async def test_reader_savings_goal_round_trip() -> None:
    """MCPLedgerReader rebuilds the SavingsGoal (incl. created_at) and net cashflow."""
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    reader = MCPLedgerReader()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"MCP {telegram_user_id}",
    )

    assert await reader.get_savings_goal(family_id=family_id) is None

    await repo.set_savings_goal(family_id=family_id, target_amount=Decimal("200000"))
    goal = await reader.get_savings_goal(family_id=family_id)

    assert goal is not None
    assert goal.target_amount == Decimal("200000")
    assert goal.created_at.tzinfo is not None

    net = await reader.net_cashflow(
        family_id=family_id,
        start=goal.created_at,
        end=datetime.now(UTC),
    )
    assert net == Decimal("0")
