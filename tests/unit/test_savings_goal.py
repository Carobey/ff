"""Unit tests for the SavingsGoal / GoalProgress domain models."""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from family_finance.domain import GoalProgress, SavingsGoal

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


def _goal(
    *,
    target: str = "100000",
    target_date: date | None = None,
    created: datetime | None = None,
) -> SavingsGoal:
    return SavingsGoal(
        family_id=uuid.uuid4(),
        target_amount=Decimal(target),
        target_date=target_date,
        created_at=created or datetime(2026, 1, 1, tzinfo=_MOSCOW),
    )


# ── SavingsGoal validation ────────────────────────────────────────────────────


@pytest.mark.unit
def test_savings_goal_rejects_nonpositive_target() -> None:
    with pytest.raises(ValidationError):
        SavingsGoal(
            family_id=uuid.uuid4(),
            target_amount=Decimal("0"),
            created_at=datetime(2026, 1, 1, tzinfo=_MOSCOW),
        )


# ── pct / remaining / reached ─────────────────────────────────────────────────


@pytest.mark.unit
def test_progress_pct_and_remaining() -> None:
    p = GoalProgress(goal=_goal(target="100000"), saved_so_far=Decimal("25000"))
    assert p.pct == 25
    assert p.remaining == Decimal("75000")
    assert p.reached is False


@pytest.mark.unit
def test_progress_reached_clamps_remaining() -> None:
    p = GoalProgress(goal=_goal(target="100000"), saved_so_far=Decimal("120000"))
    assert p.reached is True
    assert p.remaining == Decimal("0")
    assert p.pct == 120


@pytest.mark.unit
def test_progress_negative_savings_floors_pct() -> None:
    p = GoalProgress(goal=_goal(target="100000"), saved_so_far=Decimal("-5000"))
    assert p.pct == 0
    assert p.remaining == Decimal("100000")


# ── months_left / monthly_needed ──────────────────────────────────────────────


@pytest.mark.unit
def test_months_left_open_ended_is_none() -> None:
    p = GoalProgress(goal=_goal(), saved_so_far=Decimal("0"))
    now = datetime(2026, 7, 1, tzinfo=_MOSCOW)
    assert p.months_left(now) is None
    assert p.monthly_needed(now) is None
    assert p.on_track(now) is None


@pytest.mark.unit
def test_monthly_needed_divides_remaining_over_months() -> None:
    goal = _goal(target="120000", target_date=date(2026, 12, 31))
    p = GoalProgress(goal=goal, saved_so_far=Decimal("0"))
    now = datetime(2026, 7, 1, tzinfo=_MOSCOW)  # 5 months until December
    assert p.months_left(now) == 5
    assert p.monthly_needed(now) == Decimal("24000")


@pytest.mark.unit
def test_monthly_needed_past_due_returns_remaining() -> None:
    goal = _goal(target="120000", target_date=date(2026, 3, 31))
    p = GoalProgress(goal=goal, saved_so_far=Decimal("20000"))
    now = datetime(2026, 7, 1, tzinfo=_MOSCOW)  # already past the date
    assert p.months_left(now) == 0
    assert p.monthly_needed(now) == Decimal("100000")


# ── on_track ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_on_track_true_when_ahead_of_pace() -> None:
    goal = _goal(target="120000", target_date=date(2026, 12, 1))
    # created Jan, target Dec → 11 months. By July, elapsed=6 → expected ≈ 65 454.
    p = GoalProgress(goal=goal, saved_so_far=Decimal("70000"))
    assert p.on_track(datetime(2026, 7, 1, tzinfo=_MOSCOW)) is True


@pytest.mark.unit
def test_on_track_false_when_behind_pace() -> None:
    goal = _goal(target="120000", target_date=date(2026, 12, 1))
    p = GoalProgress(goal=goal, saved_so_far=Decimal("50000"))
    assert p.on_track(datetime(2026, 7, 1, tzinfo=_MOSCOW)) is False
