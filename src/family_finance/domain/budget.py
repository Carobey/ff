"""Monthly per-category budget + computed status."""

from __future__ import annotations

import uuid
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from family_finance.domain.types import Category


class Budget(BaseModel):
    """A monthly soft limit for one category inside one family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_id: uuid.UUID
    category: Category
    monthly_limit: Decimal = Field(..., gt=0)


class BudgetStatus(BaseModel):
    """Budget + how much of it has been spent this month so far.

    Returned by :meth:`PostgresTransactionRepository.get_budget_status` and
    consumed by the budgets agent / categorizer alerts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: Budget
    spent_this_month: Decimal = Field(..., ge=0)

    @property
    def pct(self) -> int:
        """Integer percentage spent of the monthly limit."""
        if self.budget.monthly_limit == Decimal("0"):
            return 0
        # Чистый Decimal без float — деньги не кастуем (PRIME-правило «no float»).
        return int((self.spent_this_month / self.budget.monthly_limit * 100).to_integral_value())

    @property
    def over_budget(self) -> bool:
        return self.spent_this_month >= self.budget.monthly_limit
