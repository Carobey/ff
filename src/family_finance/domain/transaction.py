"""
Доменная сущность Transaction.

Битемпоральная: occurred_at (когда произошло) + ingested_at (когда узнали).
Это согласуется с моделью Graphiti (t_valid/t_invalid) для Phase 2.

ПРАВИЛО: amount всегда положительное, знак выражается через direction.
Это упрощает агрегации и не даёт перепутать знак на парсинге CSV.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from family_finance.domain.types import (
    Category,
    Currency,
    Direction,
    TransactionSource,
    require_tz_aware,
)


class Transaction(BaseModel):
    """Финансовая транзакция."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,  # переоценка needs_review при правке confidence
    )

    # Идентификация
    transaction_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    family_id: uuid.UUID
    member_id: uuid.UUID

    # Битемпоральность
    occurred_at: datetime = Field(..., description="Когда транзакция реально произошла")
    posted_at: date | None = Field(
        None,
        description="Дата платежа/проводки из банковской выписки, если банк её отдаёт",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Когда мы об этом узнали",
    )

    # Деньги: amount всегда > 0, знак — в direction
    amount: Decimal = Field(..., gt=0, description="Положительная сумма")
    currency: Currency = Currency.RUB
    direction: Direction

    # Что/где
    merchant_raw: str = Field(
        ...,
        min_length=1,
        description="Сырое описание из выписки — не удалять, нужно для ре-категоризации",
    )
    merchant_normalized: str | None = None

    # Категория
    category: Category = Category.UNCLASSIFIED
    subcategory_freetext: str | None = None
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Уверенность категоризатора. <0.7 → needs_review автоматически",
    )
    needs_review: bool = Field(False)

    # Источник
    source: TransactionSource
    source_file: str | None = Field(
        None,
        description="Путь к CSV или Telegram file_id фотографии",
    )

    # Связь с чеком (опционально)
    receipt_id: uuid.UUID | None = None
    receipt_fns_qr: str | None = Field(
        None,
        description="Сырая строка QR-кода: 't=...&s=...&fn=...&i=...&fp=...&n=...'",
    )

    # Свободные теги
    tags: set[str] = Field(default_factory=set)

    # Идемпотентность импорта
    import_hash: str | None = Field(
        None,
        description=(
            "SHA256 от (occurred_at, amount, merchant_raw) — защита от дублей при ре-импорте"
        ),
    )

    @field_validator("amount", mode="before")
    @classmethod
    def to_decimal_amount(cls, v: object) -> Decimal:
        # Принимаем int/str/float, приводим к Decimal через str без потерь
        return Decimal(str(v))

    @field_validator("occurred_at", "ingested_at", mode="after")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        return require_tz_aware(v)

    @model_validator(mode="after")
    def derive_needs_review(self) -> Self:
        """needs_review полностью выводится из confidence (симметрично).

        <0.7 → требует ручной проверки; при росте уверенности (ре-категоризация)
        флаг снимается, иначе транзакция навсегда застрянет в human-in-the-loop.
        """
        # object.__setattr__ обходит validate_assignment — иначе присваивание
        # внутри model_validator снова запустит валидатор → бесконечная рекурсия.
        object.__setattr__(self, "needs_review", self.confidence < 0.7)
        return self
