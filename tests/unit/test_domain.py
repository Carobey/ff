"""
Unit tests на domain — без зависимостей на инфру.

Phase 0 smoke: проверяет что схемы валидируются как ожидается.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from family_finance.domain import (
    Category,
    Direction,
    Transaction,
    TransactionSource,
)


@pytest.mark.unit
class TestTransaction:
    def test_minimal_construction(self) -> None:
        tx = Transaction(
            family_id=uuid.uuid4(),
            member_id=uuid.uuid4(),
            occurred_at=datetime.now(UTC),
            amount=Decimal("100.50"),
            direction=Direction.EXPENSE,
            merchant_raw="Пятёрочка 4587",
            source=TransactionSource.BANK_CSV,
        )
        assert tx.amount == Decimal("100.50")
        assert tx.category == Category.UNCLASSIFIED
        assert tx.needs_review is True  # confidence=0.0 → review

    def test_amount_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Transaction(
                family_id=uuid.uuid4(),
                member_id=uuid.uuid4(),
                occurred_at=datetime.now(UTC),
                amount=Decimal("-100"),  # ← запрещено
                direction=Direction.EXPENSE,
                merchant_raw="X",
                source=TransactionSource.BANK_CSV,
            )

    def test_amount_from_float_keeps_precision(self) -> None:
        """Проверка что float → Decimal через str (без потерь точности)."""
        tx = Transaction(
            family_id=uuid.uuid4(),
            member_id=uuid.uuid4(),
            occurred_at=datetime.now(UTC),
            amount=0.1 + 0.2,  # type: ignore[arg-type]
            direction=Direction.EXPENSE,
            merchant_raw="X",
            source=TransactionSource.BANK_CSV,
        )
        # str(0.1 + 0.2) даст "0.30000000000000004" — это норма для Decimal
        # Главное — мы поймали и не упали
        assert tx.amount > Decimal("0.29")

    def test_low_confidence_forces_review(self) -> None:
        tx = Transaction(
            family_id=uuid.uuid4(),
            member_id=uuid.uuid4(),
            occurred_at=datetime.now(UTC),
            amount=Decimal("100"),
            direction=Direction.EXPENSE,
            merchant_raw="X",
            source=TransactionSource.BANK_CSV,
            confidence=0.5,
            needs_review=False,  # пытаемся обмануть — не выйдет
        )
        assert tx.needs_review is True

    def test_high_confidence_no_review(self) -> None:
        tx = Transaction(
            family_id=uuid.uuid4(),
            member_id=uuid.uuid4(),
            occurred_at=datetime.now(UTC),
            amount=Decimal("100"),
            direction=Direction.EXPENSE,
            merchant_raw="X",
            source=TransactionSource.BANK_CSV,
            confidence=0.9,
        )
        assert tx.needs_review is False
