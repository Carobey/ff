"""Доменная сущность Subscription.

Подписка / повторяющийся платёж. Не хранится в БД — детектируется на лету
агрегатами по таблице `transaction`. Это value object, который путешествует
между repo (детект) → agent (форматирование) → bot (рендер).

Критерии «это подписка»:
  * >=3 транзакций к одному продавцу за последние lookback_days
  * средняя каденция между транзакциями 20..45 дней (≈ месяц)
  * direction = EXPENSE

Variable-price подписки (например, мобильник) допускаются — мы храним
average_amount и last_amount, чтобы агент мог сказать «обычно 800, в этот
раз 1200».
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from family_finance.domain.types import Category, require_tz_aware


class Subscription(BaseModel):
    """Детектированная повторяющаяся трата."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    merchant: str = Field(..., min_length=1)
    category: Category
    cadence_days: int = Field(..., ge=1, description="Средний интервал между списаниями")
    average_amount: Decimal = Field(..., gt=0)
    last_amount: Decimal = Field(..., gt=0)
    last_seen: datetime
    occurrences: int = Field(..., ge=2)

    @field_validator("last_seen", mode="after")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        return require_tz_aware(v)
