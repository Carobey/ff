"""FastMCP server exposing read-only family-finance queries as MCP tools.

This is an *interface* layer (like ``bot/``): it imports infrastructure and
turns the existing repository read-methods into MCP tools so that any MCP
client — Claude Desktop, an external Family-Hub agent, or our own LangGraph
``ledger`` node via ``langchain-mcp-adapters`` — can query the family ledger
without touching the database directly.

Read-only by design: no tool writes to the DB. Money is returned as strings
(the repo already casts NUMERIC→TEXT) so no float ever crosses the wire.

⚠️ Trust boundary: ``family_id`` is a caller-supplied parameter and is NOT
authorized here — any client can read any family's data by passing a different
UUID. This server is therefore **single-trust / local-only**: a child stdio
process of the trusted bot (one family, local machine), not a networked
multi-tenant endpoint. Binding family to an authenticated session is post-diploma
A2A work (see docs/SECURITY.md → "Граница доверия MCP-сервера").

Run over stdio:  ``just mcp``  (or ``python -m family_finance.mcp_server.server``)
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastmcp import FastMCP

from family_finance.domain import Category, Direction, GoalProgress
from family_finance.infrastructure.persistence import PostgresTransactionRepository

mcp: FastMCP = FastMCP("family-finance")

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


def _repo() -> PostgresTransactionRepository:
    return PostgresTransactionRepository()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@mcp.tool
async def query_aggregates(
    family_id: str,
    group_by: Literal["day", "week", "month", "category", "merchant", "total"] = "total",
    then_by: Literal["day", "week", "month", "category", "merchant"] | None = None,
    categories: list[str] | None = None,
    directions: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Flexible grouped sums — the main spending-query tool.

    ``group_by`` chooses the dimension:
    - ``total`` — one grand total for the whole filter (one row).
    - ``day`` / ``week`` / ``month`` — time breakdown (Europe/Moscow calendar),
      chronological. Use this for «по дням / помесячно».
    - ``category`` / ``merchant`` — breakdown by category or merchant, biggest
      first.

    ``then_by`` adds a second dimension for a 2-D breakdown such as «по дням по
    категориям» (``group_by="day", then_by="category"``); each row then also
    carries a ``subbucket``.

    ``categories``/``directions`` are domain enum values (e.g. ``food.groceries``,
    ``expense``); omit or pass ``[]`` for "all". ``start``/``end`` are ISO-8601
    datetimes; omit for all-time. Returns a list of
    ``{"bucket", "subbucket", "total", "count"}`` where ``bucket`` is the group
    key as a string (date, ``YYYY-MM``, category value or merchant name) and
    ``subbucket`` is ``None`` unless ``then_by`` is set.
    """
    buckets = await _repo().query_aggregates(
        family_id=uuid.UUID(family_id),
        group_by=group_by,
        then_by=then_by,
        categories=[Category(c) for c in (categories or [])],
        directions=[Direction(d) for d in (directions or [])],
        start=_parse_dt(start),
        end=_parse_dt(end),
        limit=limit,
    )
    return [
        {"bucket": b.bucket, "subbucket": b.subbucket, "total": str(b.total), "count": b.count}
        for b in buckets
    ]


@mcp.tool
async def list_transactions(
    family_id: str,
    categories: list[str] | None = None,
    directions: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    order_by: Literal["date_desc", "amount_desc"] = "date_desc",
    limit: int = 20,
) -> list[dict[str, object]]:
    """List individual transactions (not aggregated).

    Use for «покажи списком», «топ-5 крупных трат». ``order_by`` is
    ``date_desc`` (newest first) or ``amount_desc`` (biggest first). Filters and
    period behave like ``query_aggregates``. Returns a list of
    ``{"occurred_at", "amount", "direction", "category", "merchant"}``.
    """
    rows = await _repo().list_transactions(
        family_id=uuid.UUID(family_id),
        categories=[Category(c) for c in (categories or [])],
        directions=[Direction(d) for d in (directions or [])],
        start=_parse_dt(start),
        end=_parse_dt(end),
        order_by=order_by,
        limit=limit,
    )
    return [
        {
            "occurred_at": e.occurred_at.isoformat(),
            "amount": str(e.amount),
            "direction": e.direction.value,
            "category": e.category.value,
            "merchant": e.merchant,
        }
        for e in rows
    ]


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
