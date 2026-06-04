"""Unit tests for Tinkoff CSV parsing."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from family_finance.domain import Category, Currency, Direction, TransactionSource
from family_finance.infrastructure.parsers import TinkoffCsvParser


@pytest.mark.unit
def test_tinkoff_parser_builds_transactions() -> None:
    family_id = uuid.uuid4()
    member_id = uuid.uuid4()
    sample = Path("tests/samples/tinkoff_sample.csv").read_bytes()

    transactions = TinkoffCsvParser().parse(
        sample,
        family_id=family_id,
        member_id=member_id,
        source_file="sample.csv",
    )

    assert len(transactions) == 2
    first = transactions[0]
    assert first.family_id == family_id
    assert first.member_id == member_id
    assert first.occurred_at == datetime(2026, 5, 3, 10, 56, 10, tzinfo=UTC)
    assert first.posted_at == date(2026, 5, 3)
    assert first.amount == Decimal("903.26")
    assert first.currency == Currency.RUB
    assert first.direction == Direction.EXPENSE
    assert first.category == Category.FOOD_GROCERIES
    assert first.confidence == 0.8
    assert first.needs_review is False
    assert first.source == TransactionSource.BANK_CSV
    assert first.import_hash is not None


@pytest.mark.unit
def test_tinkoff_parser_rejects_unknown_headers() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        TinkoffCsvParser().parse(
            b"bad;csv\n1;2\n",
            family_id=uuid.uuid4(),
            member_id=uuid.uuid4(),
        )


@pytest.mark.unit
def test_tinkoff_parser_marks_transfers_for_review() -> None:
    content = (
        '"Дата операции";"Статус";"Сумма платежа";"Валюта платежа";"Категория";"Описание"\n'
        '"03.05.2026 13:56:10";"OK";"-1000,00";"RUB";"Переводы";"Ivan Ivanov"\n'
    ).encode()

    transactions = TinkoffCsvParser().parse(
        content,
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
    )

    assert len(transactions) == 1
    assert transactions[0].direction == Direction.TRANSFER
    assert transactions[0].category == Category.TRANSFER_INTERNAL
    assert transactions[0].confidence == 0.6
    assert transactions[0].needs_review is True
