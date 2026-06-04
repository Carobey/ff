"""Unit tests for ReceiptAgent helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from family_finance.agents.receipt import _categorise_item, _items_word, receipt_to_transactions
from family_finance.domain import Category, Direction, TransactionSource
from family_finance.domain.receipt import Receipt, ReceiptItem

_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_NOW = datetime(2026, 5, 2, 12, 34, 56, tzinfo=UTC)


# ── _categorise_item ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_categorise_item_milk() -> None:
    assert _categorise_item("Молоко 3.2%") == Category.FOOD_GROCERIES


@pytest.mark.unit
def test_categorise_item_aspirin() -> None:
    assert _categorise_item("Аспирин таблетки 10шт") == Category.HEALTH_PHARMACY


@pytest.mark.unit
def test_categorise_item_diaper() -> None:
    assert _categorise_item("Подгузники Pampers 44шт") == Category.KIDS_TOYS


@pytest.mark.unit
def test_categorise_item_notebook() -> None:
    assert _categorise_item("Тетрадь школьная 12л") == Category.KIDS_SCHOOL


@pytest.mark.unit
def test_categorise_item_unknown_defaults_to_groceries() -> None:
    assert _categorise_item("Неизвестный товар XYZ") == Category.FOOD_GROCERIES


# ── _items_word ───────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "позиция"),
        (2, "позиции"),
        (5, "позиций"),
        (11, "позиций"),  # 11..19 always "позиций"
        (21, "позиция"),
        (100, "позиций"),
    ],
)
def test_items_word(n: int, expected: str) -> None:
    assert _items_word(n) == expected


# ── receipt_to_transactions ───────────────────────────────────────────────────


def _make_receipt(items: list[ReceiptItem], total: Decimal = Decimal("500.00")) -> Receipt:
    return Receipt(
        family_id=_FAMILY_ID,
        member_id=_MEMBER_ID,
        qr_raw="t=20260502T1234&s=500.00&fn=111&i=222&fp=333&n=1",
        total_amount=total,
        purchase_time=_NOW,
        store_name="Пятёрочка",
        items=items,
    )


@pytest.mark.unit
def test_receipt_to_transactions_with_items() -> None:
    items = [
        ReceiptItem(
            name="Молоко", quantity=Decimal("1"), price=Decimal("89.90"), total=Decimal("89.90")
        ),
        ReceiptItem(
            name="Хлеб", quantity=Decimal("1"), price=Decimal("45.00"), total=Decimal("45.00")
        ),
    ]
    receipt = _make_receipt(items)
    txs = receipt_to_transactions(receipt, family_id=_FAMILY_ID, member_id=_MEMBER_ID)

    assert len(txs) == 2
    assert all(tx.direction == Direction.EXPENSE for tx in txs)
    assert all(tx.source == TransactionSource.RECEIPT_PHOTO_QR for tx in txs)
    assert txs[0].amount == Decimal("89.90")
    assert txs[1].amount == Decimal("45.00")
    # merchant_raw should include store name
    assert "Пятёрочка" in txs[0].merchant_raw
    # categories from keyword matching
    assert txs[0].category == Category.FOOD_GROCERIES


@pytest.mark.unit
def test_receipt_to_transactions_no_items_uses_total() -> None:
    receipt = _make_receipt(items=[], total=Decimal("1234.56"))
    txs = receipt_to_transactions(receipt, family_id=_FAMILY_ID, member_id=_MEMBER_ID)

    assert len(txs) == 1
    assert txs[0].amount == Decimal("1234.56")
    assert txs[0].source == TransactionSource.RECEIPT_PHOTO_QR


@pytest.mark.unit
def test_receipt_to_transactions_idempotent_import_hash() -> None:
    """Same receipt → same import_hash → deduplication works."""
    items = [
        ReceiptItem(name="Сыр", quantity=Decimal("1"), price=Decimal("300"), total=Decimal("300")),
    ]
    receipt = _make_receipt(items)
    txs1 = receipt_to_transactions(receipt, family_id=_FAMILY_ID, member_id=_MEMBER_ID)
    txs2 = receipt_to_transactions(receipt, family_id=_FAMILY_ID, member_id=_MEMBER_ID)
    assert txs1[0].import_hash == txs2[0].import_hash


@pytest.mark.unit
def test_receipt_to_transactions_amount_always_positive() -> None:
    items = [
        ReceiptItem(name="Кефир", quantity=Decimal("2"), price=Decimal("60"), total=Decimal("120")),
    ]
    receipt = _make_receipt(items)
    txs = receipt_to_transactions(receipt, family_id=_FAMILY_ID, member_id=_MEMBER_ID)
    for tx in txs:
        assert tx.amount > 0


# ── ProverkaCheka datetime parser ─────────────────────────────────────────────


@pytest.mark.unit
def test_parse_datetime_fiscal_compact() -> None:
    from family_finance.infrastructure.parsers.proverkacheka import _parse_datetime

    dt = _parse_datetime("20260502T123456")
    assert dt.year == 2026
    assert dt.month == 5
    assert dt.day == 2
    assert dt.hour == 12
    assert dt.minute == 34


@pytest.mark.unit
def test_parse_datetime_iso() -> None:
    from family_finance.infrastructure.parsers.proverkacheka import _parse_datetime

    dt = _parse_datetime("2026-05-02T12:34:00")
    assert dt.year == 2026


@pytest.mark.unit
def test_parse_datetime_empty_returns_now() -> None:
    from family_finance.infrastructure.parsers.proverkacheka import _parse_datetime

    dt = _parse_datetime("")
    assert dt.tzinfo is not None
