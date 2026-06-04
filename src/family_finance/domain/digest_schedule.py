"""Domain model for the user-configurable weekly-digest schedule.

The actual triggering happens via APScheduler's ``CronTrigger`` — we keep
the on-disk representation minimal (a single string the scheduler can
parse) and use this value object as the in-memory contract between the
NLP parser, repo, and scheduler.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_ALLOWED_DAYS: tuple[DayOfWeek, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class DigestSchedule(BaseModel):
    """When to push the weekly digest into the user's chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day_of_week: DayOfWeek
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(0, ge=0, le=59)
    # We deliberately don't expose per-user timezones in v1 — everything
    # runs in Europe/Moscow because that's where the families live.

    def to_cron(self) -> str:
        """Render as a 5-field cron expression compatible with APScheduler.

        Format: ``minute hour day-of-month month day-of-week``.
        """
        return f"{self.minute} {self.hour} * * {self.day_of_week}"

    @classmethod
    def from_cron(cls, cron: str) -> DigestSchedule:
        """Parse the value produced by :meth:`to_cron`. Strict on shape."""
        parts = cron.split()
        if len(parts) != 5:
            msg = f"Invalid cron (expected 5 fields, got {len(parts)}): {cron!r}"
            raise ValueError(msg)
        minute_s, hour_s, _dom, _mon, dow = parts
        if dow not in _ALLOWED_DAYS:
            msg = f"Invalid day-of-week: {dow!r}"
            raise ValueError(msg)
        try:
            hour, minute = int(hour_s), int(minute_s)
        except ValueError as exc:
            msg = f"Invalid cron time fields (hour={hour_s!r}, minute={minute_s!r}): {cron!r}"
            raise ValueError(msg) from exc
        return cls(day_of_week=dow, hour=hour, minute=minute)

    def human_label(self) -> str:
        """Render a Russian label for the user — used in confirmation replies."""
        return f"{_DAY_LABELS_RU[self.day_of_week]} в {self.hour:02d}:{self.minute:02d}"


_DAY_LABELS_RU: dict[str, str] = {
    "mon": "по понедельникам",
    "tue": "по вторникам",
    "wed": "по средам",
    "thu": "по четвергам",
    "fri": "по пятницам",
    "sat": "по субботам",
    "sun": "по воскресеньям",
}
