"""Read-side adapter that serves ledger data through the MCP server.

The advisor / budgets / subscriptions LangGraph nodes used to call
``PostgresTransactionRepository`` directly. They now go through this adapter,
which invokes the family-finance MCP tools and rebuilds the same domain objects
the repository would have returned. The method signatures mirror the repo's
read interface so the nodes (and their injected-``repo`` helpers) swap in
without other changes.

Money crosses the MCP boundary as strings; we parse straight back to
``Decimal`` here — no float ever appears.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from family_finance.application.ports import LedgerSummary
from family_finance.domain import (
    Budget,
    BudgetStatus,
    Category,
    Direction,
    SavingsGoal,
    Subscription,
)
from family_finance.infrastructure.mcp.client import call_finance_tool


class MCPLedgerReader:
    """Repository-shaped read facade backed by the family-finance MCP tools."""

    async def aggregate(
        self,
        *,
        family_id: uuid.UUID,
        categories: Sequence[Category],
        directions: Sequence[Direction],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> LedgerSummary:
        data = await call_finance_tool(
            "aggregate_spending",
            {
                "family_id": str(family_id),
                "categories": [c.value for c in categories],
                "directions": [d.value for d in directions],
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
        )
        return LedgerSummary(total=Decimal(str(data["total"])), count=int(data["count"]))

    async def category_breakdown(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> list[tuple[Category, Decimal, int]]:
        rows = await call_finance_tool(
            "spending_by_category",
            {"family_id": str(family_id), "start": start.isoformat(), "end": end.isoformat()},
        )
        return [(Category(r["category"]), Decimal(str(r["total"])), int(r["count"])) for r in rows]

    async def get_savings_goal(self, *, family_id: uuid.UUID) -> SavingsGoal | None:
        data = await call_finance_tool("savings_goal", {"family_id": str(family_id)})
        if not data.get("has_goal"):
            return None
        target_date_raw = data.get("target_date")
        return SavingsGoal(
            family_id=family_id,
            target_amount=Decimal(str(data["target_amount"])),
            target_date=date.fromisoformat(target_date_raw) if target_date_raw else None,
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )

    async def net_cashflow(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> Decimal:
        data = await call_finance_tool(
            "net_cashflow",
            {"family_id": str(family_id), "start": start.isoformat(), "end": end.isoformat()},
        )
        return Decimal(str(data["net"]))

    async def get_budget_status(
        self,
        *,
        family_id: uuid.UUID,
        month_start: datetime,
        month_end: datetime,
    ) -> list[BudgetStatus]:
        rows = await call_finance_tool(
            "budget_status",
            {
                "family_id": str(family_id),
                "month_start": month_start.isoformat(),
                "month_end": month_end.isoformat(),
            },
        )
        return [
            BudgetStatus(
                budget=Budget(
                    family_id=family_id,
                    category=Category(r["category"]),
                    monthly_limit=Decimal(str(r["monthly_limit"])),
                ),
                spent_this_month=Decimal(str(r["spent_this_month"])),
            )
            for r in rows
        ]

    async def detect_recurring(self, *, family_id: uuid.UUID) -> list[Subscription]:
        rows = await call_finance_tool("detect_recurring", {"family_id": str(family_id)})
        return [
            Subscription(
                merchant=r["merchant"],
                category=Category(r["category"]),
                cadence_days=int(r["cadence_days"]),
                average_amount=Decimal(str(r["average_amount"])),
                last_amount=Decimal(str(r["last_amount"])),
                last_seen=datetime.fromisoformat(str(r["last_seen"])),
                occurrences=int(r["occurrences"]),
            )
            for r in rows
        ]
