"""Доменная сущность Receipt + позиции чека."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from family_finance.domain.types import Category, require_tz_aware


class ReceiptItem(BaseModel):
    """Одна позиция из детализации чека (ФНС API возвращает массив таких)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Наименование товара из чека ОФД")
    quantity: Decimal = Field(..., gt=0)
    price: Decimal = Field(..., description="Цена за единицу")
    total: Decimal = Field(..., description="Итого = quantity * price")
    nds_amount: Decimal | None = Field(
        None,
        description="НДС — для налоговых вычетов",
    )
    predicted_category: Category | None = None

    @field_validator("quantity", "price", "total", "nds_amount", mode="before")
    @classmethod
    def to_decimal(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        return Decimal(str(v))

    @model_validator(mode="after")
    def _check_total(self) -> Self:
        """Целостность позиции: total ≈ quantity*price (допуск на округление ОФД)."""
        expected = self.quantity * self.price
        if abs(self.total - expected) > Decimal("0.01"):
            msg = f"ReceiptItem.total {self.total} != quantity*price {expected}"
            raise ValueError(msg)
        return self


class Receipt(BaseModel):
    """Чек целиком (FNS detail response + наши метаданные)."""

    model_config = ConfigDict(extra="forbid")

    receipt_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    family_id: uuid.UUID
    member_id: uuid.UUID

    qr_raw: str = Field(
        ...,
        description="Строка из QR: 't=...&s=...&fn=...&i=...&fp=...&n=1'",
    )

    # Фискальные реквизиты
    fiscal_drive: str | None = None  # FN
    fiscal_document: str | None = None  # FD/i
    fiscal_sign: str | None = None  # FP

    total_amount: Decimal = Field(
        ...,
        description=(
            "Итоговая сумма как её прислал ОФД (totalSum). Авторитетное значение — "
            "НЕ пересчитываем из items: часть позиций может не распарситься, плюс скидки."
        ),
    )
    purchase_time: datetime
    store_name: str | None = None

    items: list[ReceiptItem] = Field(default_factory=list)
    raw_response: dict[str, object] | None = Field(
        None,
        description="Полный ответ ФНС/ОФД API для отладки",
    )

    @field_validator("total_amount", mode="before")
    @classmethod
    def to_decimal(cls, v: object) -> Decimal:
        return Decimal(str(v))

    @field_validator("purchase_time", mode="after")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        return require_tz_aware(v)
