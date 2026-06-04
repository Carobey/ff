"""A family's savings goal + computed progress (pay-yourself-first)."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from family_finance.domain.types import require_tz_aware


class SavingsGoal(BaseModel):
    """A single savings target for a family.

    One goal per family. ``target_date`` is optional — without it the goal is
    open-ended (progress only, no "by when" pacing). ``created_at`` anchors the
    window over which we measure net savings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_id: uuid.UUID
    target_amount: Decimal = Field(..., gt=0)
    target_date: date | None = None
    created_at: datetime

    @field_validator("created_at", mode="after")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        return require_tz_aware(v)


class GoalProgress(BaseModel):
    """Goal + how much has been saved toward it so far.

    ``saved_so_far`` is net cashflow (income − expenses) since the goal was
    set; it can be negative when a family outspent its income, so we don't
    constrain it. The display percentage floors at 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: SavingsGoal
    saved_so_far: Decimal

    @property
    def pct(self) -> int:
        """Integer percentage of the target saved, floored at 0."""
        # Чистый Decimal без float — деньги не кастуем (PRIME-правило «no float»).
        raw = int((self.saved_so_far / self.goal.target_amount * 100).to_integral_value())
        return max(raw, 0)

    @property
    def remaining(self) -> Decimal:
        """How much is still left to save (clamped to [0, target])."""
        saved = self.saved_so_far if self.saved_so_far > 0 else Decimal("0")
        left = self.goal.target_amount - saved
        return left if left > 0 else Decimal("0")

    @property
    def reached(self) -> bool:
        return self.saved_so_far >= self.goal.target_amount

    def months_left(self, now: datetime) -> int | None:
        """Whole months from *now* until ``target_date`` (None if open-ended)."""
        if self.goal.target_date is None:
            return None
        target = self.goal.target_date
        months = (target.year - now.year) * 12 + (target.month - now.month)
        return max(months, 0)

    def monthly_needed(self, now: datetime) -> Decimal | None:
        """How much to set aside per month to hit the goal on time."""
        months = self.months_left(now)
        if months is None:
            return None
        if months == 0:
            return self.remaining
        return (self.remaining / months).quantize(Decimal("1"))

    def on_track(self, now: datetime) -> bool | None:
        """Whether saved-so-far meets the linear pace toward ``target_date``.

        None when the goal is open-ended (no date to pace against).
        """
        if self.goal.target_date is None:
            return None
        if self.reached:
            return True
        total_months = (self.goal.target_date.year - self.goal.created_at.year) * 12 + (
            self.goal.target_date.month - self.goal.created_at.month
        )
        if total_months <= 0:
            return self.reached
        elapsed = (now.year - self.goal.created_at.year) * 12 + (
            now.month - self.goal.created_at.month
        )
        elapsed = min(max(elapsed, 0), total_months)
        expected = self.goal.target_amount * Decimal(elapsed) / Decimal(total_months)
        return self.saved_so_far >= expected
