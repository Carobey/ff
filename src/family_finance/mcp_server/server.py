"""FastMCP server exposing read-only family-finance queries as MCP tools.

This is an *interface* layer (like ``bot/``): it imports infrastructure and
turns the existing repository read-methods into MCP tools so that any MCP
client — Claude Desktop, an external Family-Hub agent, or our own LangGraph
``ledger`` node via ``langchain-mcp-adapters`` — can query the family ledger
without touching the database directly.

Read-only by design: no tool writes to the DB. Money is returned as strings
(the repo already casts NUMERIC→TEXT) so no float ever crosses the wire.

Run over stdio:  ``just mcp``  (or ``python -m family_finance.mcp_server.server``)
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal

from fastmcp import FastMCP

from family_finance.domain import Category, Direction, GoalProgress
from family_finance.infrastructure.persistence import PostgresTransactionRepository

mcp: FastMCP = FastMCP("family-finance")

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


def _repo() -> PostgresTransactionRepository:
    return PostgresTransactionRepository()


def _month_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Current Moscow calendar month [start, next-month-start)."""
    now = now or datetime.now(_MOSCOW)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + (start.month // 12), month=(start.month % 12) + 1)
    return start, end


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@mcp.tool
async def aggregate_spending(
    family_id: str,
    categories: list[str],
    directions: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, object]:
    """Sum transactions for a family, scoped by categories/directions/period.

    ``categories``/``directions`` are domain enum values (e.g. ``food.groceries``,
    ``expense``). ``start``/``end`` are ISO-8601 datetimes; omit for all-time.
    Returns ``{"total": "<rubles as string>", "count": <int>}``.
    """
    summary = await _repo().aggregate(
        family_id=uuid.UUID(family_id),
        categories=[Category(c) for c in categories],
        directions=[Direction(d) for d in directions],
        start=_parse_dt(start),
        end=_parse_dt(end),
    )
    return {"total": str(summary.total), "count": summary.count}


@mcp.tool
async def spending_by_category(
    family_id: str,
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, object]]:
    """Per-category expense totals for a period, biggest first.

    Defaults to the current Moscow calendar month when ``start``/``end`` are
    omitted. Returns a list of ``{"category", "total", "count"}``.
    """
    win_start = _parse_dt(start)
    win_end = _parse_dt(end)
    if win_start is None or win_end is None:
        win_start, win_end = _month_window()
    rows = await _repo().category_breakdown(
        family_id=uuid.UUID(family_id), start=win_start, end=win_end
    )
    return [{"category": cat.value, "total": str(amount), "count": n} for cat, amount, n in rows]


@mcp.tool
async def goal_status(family_id: str) -> dict[str, object]:
    """Savings-goal progress for a family.

    Returns ``{"has_goal": False}`` when no goal is set, otherwise target,
    saved-so-far (net cashflow since the goal was created) and percentage.
    """
    repo = _repo()
    fam = uuid.UUID(family_id)
    goal = await repo.get_savings_goal(family_id=fam)
    if goal is None:
        return {"has_goal": False}
    now = datetime.now(_MOSCOW)
    saved = await repo.net_cashflow(family_id=fam, start=goal.created_at, end=now)
    progress = GoalProgress(goal=goal, saved_so_far=saved)
    return {
        "has_goal": True,
        "target_amount": str(goal.target_amount),
        "saved_so_far": str(saved if saved > Decimal("0") else Decimal("0")),
        "remaining": str(progress.remaining),
        "pct": progress.pct,
        "reached": progress.reached,
        "target_date": goal.target_date.isoformat() if goal.target_date else None,
    }


@mcp.tool
async def savings_goal(family_id: str) -> dict[str, object]:
    """Raw savings goal for a family (no progress computation).

    Returns ``{"has_goal": False}`` when none is set, otherwise the stored
    target, optional ``target_date`` and the ``created_at`` anchor used to
    measure net savings. Use ``net_cashflow`` to compute progress.
    """
    goal = await _repo().get_savings_goal(family_id=uuid.UUID(family_id))
    if goal is None:
        return {"has_goal": False}
    return {
        "has_goal": True,
        "target_amount": str(goal.target_amount),
        "target_date": goal.target_date.isoformat() if goal.target_date else None,
        "created_at": goal.created_at.isoformat(),
    }


@mcp.tool
async def net_cashflow(family_id: str, start: str, end: str) -> dict[str, object]:
    """Net savings over ``[start, end)``: income + refunds − expenses.

    Internal transfers are ignored. ``start``/``end`` are ISO-8601 datetimes.
    Returns ``{"net": "<rubles as string>"}``.
    """
    net = await _repo().net_cashflow(
        family_id=uuid.UUID(family_id),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
    )
    return {"net": str(net)}


@mcp.tool
async def budget_status(
    family_id: str,
    month_start: str,
    month_end: str,
) -> list[dict[str, object]]:
    """Per-category budget usage over ``[month_start, month_end)``.

    Returns one row per configured budget:
    ``{"category", "monthly_limit", "spent_this_month"}`` (amounts as strings).
    """
    statuses = await _repo().get_budget_status(
        family_id=uuid.UUID(family_id),
        month_start=datetime.fromisoformat(month_start),
        month_end=datetime.fromisoformat(month_end),
    )
    return [
        {
            "category": s.budget.category.value,
            "monthly_limit": str(s.budget.monthly_limit),
            "spent_this_month": str(s.spent_this_month),
        }
        for s in statuses
    ]


@mcp.tool
async def detect_recurring(family_id: str) -> list[dict[str, object]]:
    """Detect recurring expenses (subscriptions, regular bills) for a family.

    Heuristic SQL aggregate over the last year, newest-seen first. Returns
    ``{"merchant", "category", "cadence_days", "average_amount", "last_amount",
    "last_seen", "occurrences"}`` per detected subscription (amounts as strings).
    """
    subs = await _repo().detect_recurring(family_id=uuid.UUID(family_id))
    return [
        {
            "merchant": s.merchant,
            "category": s.category.value,
            "cadence_days": s.cadence_days,
            "average_amount": str(s.average_amount),
            "last_amount": str(s.last_amount),
            "last_seen": s.last_seen.isoformat(),
            "occurrences": s.occurrences,
        }
        for s in subs
    ]


if __name__ == "__main__":
    mcp.run()
