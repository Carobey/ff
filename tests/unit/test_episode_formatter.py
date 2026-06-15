"""Unit tests for episodic memory episode formatter."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from family_finance.domain import Category, Currency, Direction, Transaction, TransactionSource
from family_finance.infrastructure.memory.episode_formatter import (
    import_to_episode_body,
    make_episode_name,
    make_import_episode_name,
    transaction_to_episode_body,
)

_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _tx(
    *,
    amount: str = "523.40",
    direction: Direction = Direction.EXPENSE,
    merchant_raw: str = "Пятёрочка 4587",
    category: Category = Category.FOOD_GROCERIES,
    occurred_at: datetime | None = None,
    import_hash: str = "abc123def456",
) -> Transaction:
    return Transaction(
        family_id=_FAMILY_ID,
        member_id=_MEMBER_ID,
        occurred_at=occurred_at or datetime(2026, 5, 29, 10, 0, tzinfo=UTC),
        amount=Decimal(amount),
        currency=Currency.RUB,
        direction=direction,
        merchant_raw=merchant_raw,
        category=category,
        confidence=0.8,
        source=TransactionSource.BANK_CSV,
        import_hash=import_hash,
    )


# ── transaction_to_episode_body ───────────────────────────────────────────────


@pytest.mark.unit
def test_expense_contains_merchant_and_date() -> None:
    body = transaction_to_episode_body(_tx())
    assert "Пятёрочка" in body
    assert "мая" in body
    assert "2026" in body


@pytest.mark.unit
def test_expense_contains_amount() -> None:
    body = transaction_to_episode_body(_tx(amount="1234.00"))
    # formatter uses non-breaking space (\xa0) as thousands separator
    assert "1\xa0234" in body
    assert "₽" in body


@pytest.mark.unit
def test_expense_contains_owner_name() -> None:
    body = transaction_to_episode_body(_tx(), owner_name="Юри")
    assert "Юри" in body


@pytest.mark.unit
def test_income_starts_with_доход() -> None:
    body = transaction_to_episode_body(
        _tx(direction=Direction.INCOME, merchant_raw="Работодатель ООО")
    )
    assert body.startswith("Доход")
    assert "Работодатель" in body


@pytest.mark.unit
def test_transfer_starts_with_перевод() -> None:
    body = transaction_to_episode_body(_tx(direction=Direction.TRANSFER))
    assert body.startswith("Перевод")


@pytest.mark.unit
def test_unknown_merchant_produces_valid_body() -> None:
    """Merchant with a single space still produces a valid episode body."""
    body = transaction_to_episode_body(_tx(merchant_raw=" "))
    # merchant.strip() == "" → falls through to "потратил(а)" branch
    assert "₽" in body
    assert len(body) > 10


@pytest.mark.unit
def test_may_month_genitive() -> None:
    """Month 5 should appear as 'мая' (genitive)."""
    tx = _tx(occurred_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC))
    body = transaction_to_episode_body(tx)
    assert "мая" in body


@pytest.mark.unit
def test_december_month_genitive() -> None:
    tx = _tx(occurred_at=datetime(2026, 12, 1, 12, 0, tzinfo=UTC))
    body = transaction_to_episode_body(tx)
    assert "декабря" in body


# ── make_episode_name ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_episode_name_uses_import_hash_prefix() -> None:
    name = make_episode_name(_tx(import_hash="abcdef123456789"))
    assert name == "tx:abcdef123456"


@pytest.mark.unit
def test_episode_name_stable() -> None:
    """Same transaction → same name on every call."""
    tx = _tx(import_hash="xyz")
    assert make_episode_name(tx) == make_episode_name(tx)


@pytest.mark.unit
def test_episode_name_starts_with_tx_prefix() -> None:
    name = make_episode_name(_tx())
    assert name.startswith("tx:")


# ── import_to_episode_body / make_import_episode_name ─────────────────────────


@pytest.mark.unit
def test_import_body_summarizes_period_total_and_categories() -> None:
    txs = [
        _tx(
            amount="1000.00",
            category=Category.FOOD_GROCERIES,
            occurred_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
            import_hash="h1",
        ),
        _tx(
            amount="500.00",
            category=Category.FOOD_GROCERIES,
            occurred_at=datetime(2026, 5, 20, 9, 0, tzinfo=UTC),
            import_hash="h2",
        ),
        _tx(
            amount="300.00",
            category=Category.FOOD_RESTAURANT,
            occurred_at=datetime(2026, 5, 31, 9, 0, tzinfo=UTC),
            import_hash="h3",
        ),
    ]
    body = import_to_episode_body(txs)
    assert "импорт выписки" in body
    assert "3 операций" in body
    # Top category (groceries 1500) ranks before restaurant (300).
    assert body.index("groceries") < body.index("restaurant")
    assert "мая 2026" in body


@pytest.mark.unit
def test_import_body_ignores_income_and_transfers() -> None:
    txs = [
        _tx(amount="1000.00", direction=Direction.EXPENSE, import_hash="h1"),
        _tx(amount="9999.00", direction=Direction.INCOME, import_hash="h2"),
        _tx(amount="5000.00", direction=Direction.TRANSFER, import_hash="h3"),
    ]
    body = import_to_episode_body(txs)
    assert "1 операций" in body
    assert "9\xa0999" not in body


@pytest.mark.unit
def test_import_body_no_expenses() -> None:
    body = import_to_episode_body([_tx(direction=Direction.TRANSFER, import_hash="h1")])
    assert "без расходных операций" in body


@pytest.mark.unit
def test_import_episode_name_stable_and_order_independent() -> None:
    a = _tx(import_hash="h1")
    b = _tx(import_hash="h2")
    name1 = make_import_episode_name([a, b])
    name2 = make_import_episode_name([b, a])
    assert name1 == name2
    assert name1.startswith("import:")
