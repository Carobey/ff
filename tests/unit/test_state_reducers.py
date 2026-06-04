"""Unit tests for the LangGraph state reducers in ``agents/state.py``.

The reducers are tiny but load-bearing: ``merge_transactions`` keeps
parallel updates from creating duplicate rows in state, and
``replace_open_questions`` keeps a single active batch of clarifications.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from family_finance.agents.state import merge_transactions, replace_open_questions
from family_finance.domain import (
    Category,
    Currency,
    Direction,
    Transaction,
    TransactionSource,
)


def _tx(
    *,
    amount: str = "100.00",
    category: Category = Category.UNCLASSIFIED,
    confidence: float = 0.4,
    transaction_id: uuid.UUID | None = None,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id or uuid.uuid4(),
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        amount=Decimal(amount),
        currency=Currency.RUB,
        direction=Direction.EXPENSE,
        merchant_raw="TEST",
        category=category,
        confidence=confidence,
        source=TransactionSource.BANK_CSV,
    )


@pytest.mark.unit
def test_merge_transactions_appends_when_disjoint() -> None:
    existing = [_tx()]
    new = [_tx()]
    merged = merge_transactions(existing, new)
    assert len(merged) == 2


@pytest.mark.unit
def test_merge_transactions_replaces_same_id() -> None:
    """A categorizer update for the same transaction_id must REPLACE, not duplicate."""
    tid = uuid.uuid4()
    original = _tx(transaction_id=tid, category=Category.UNCLASSIFIED, confidence=0.3)
    enriched = _tx(transaction_id=tid, category=Category.FOOD_GROCERIES, confidence=0.9)

    merged = merge_transactions([original], [enriched])

    assert len(merged) == 1
    assert merged[0].category == Category.FOOD_GROCERIES
    assert merged[0].confidence == 0.9


@pytest.mark.unit
def test_merge_transactions_handles_none_existing() -> None:
    new = [_tx()]
    assert merge_transactions(None, new) == new


@pytest.mark.unit
def test_replace_open_questions_overwrites() -> None:
    """Clarifications come in batches — new batch replaces the old one."""
    old: list = [{"id": 1, "text": "stale"}]
    new: list = [{"id": 1, "text": "fresh"}, {"id": 2, "text": "another"}]
    assert replace_open_questions(old, new) == new


@pytest.mark.unit
def test_replace_open_questions_clears_with_empty_list() -> None:
    old: list = [{"id": 1, "text": "stale"}]
    assert replace_open_questions(old, []) == []
