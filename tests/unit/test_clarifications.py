"""Unit tests for import clarification questions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from family_finance.agents.clarifications import (
    build_import_questions,
    format_import_questions,
    parse_clarification_answers,
)
from family_finance.domain import Category, Currency, Direction, Transaction, TransactionSource


def _tx(
    *,
    merchant_raw: str,
    amount: str,
    direction: Direction,
    category: Category,
    confidence: float,
) -> Transaction:
    return Transaction(
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 4, 1, tzinfo=UTC),
        amount=Decimal(amount),
        currency=Currency.RUB,
        direction=direction,
        merchant_raw=merchant_raw,
        category=category,
        confidence=confidence,
        source=TransactionSource.BANK_CSV,
    )


@pytest.mark.unit
def test_build_import_questions_for_transfer_and_unknown_category() -> None:
    questions = format_import_questions(
        build_import_questions(
            [
                _tx(
                    merchant_raw="Ivan Ivanov",
                    amount="1000",
                    direction=Direction.TRANSFER,
                    category=Category.TRANSFER_INTERNAL,
                    confidence=0.6,
                ),
                _tx(
                    merchant_raw="Unknown Shop",
                    amount="500",
                    direction=Direction.EXPENSE,
                    category=Category.UNCLASSIFIED,
                    confidence=0.0,
                ),
            ],
        )
    )

    assert questions == [
        "Непонятная категория: 01.04.2026, «Unknown Shop», 500.00 ₽. Что это?",
        "Перевод: 01.04.2026, «Ivan Ivanov», 1000.00 ₽. Что это?",
    ]


@pytest.mark.unit
def test_build_import_questions_keeps_import_hashes() -> None:
    transaction = _tx(
        merchant_raw="Unknown Shop",
        amount="500",
        direction=Direction.EXPENSE,
        category=Category.UNCLASSIFIED,
        confidence=0.0,
    )
    transaction.import_hash = "hash-1"

    questions = build_import_questions([transaction])

    assert questions[0]["import_hashes"] == ["hash-1"]


@pytest.mark.unit
def test_parse_clarification_answers_maps_numbered_reply() -> None:
    transactions = [
        _tx(
            merchant_raw="CONCEPT CLUB",
            amount="4607",
            direction=Direction.EXPENSE,
            category=Category.UNCLASSIFIED,
            confidence=0.0,
        ),
        _tx(
            merchant_raw="EKOPROMLIPETSK",
            amount="1281.60",
            direction=Direction.EXPENSE,
            category=Category.UNCLASSIFIED,
            confidence=0.0,
        ),
        _tx(
            merchant_raw="GGsel.net",
            amount="1639",
            direction=Direction.EXPENSE,
            category=Category.UNCLASSIFIED,
            confidence=0.0,
        ),
    ]
    for index, transaction in enumerate(transactions, start=1):
        transaction.import_hash = f"hash-{index}"
    questions = build_import_questions(transactions)

    answers = parse_clarification_answers(
        "1 одежда 2 оплата комунальных услуг вывоз мусора 3 подписки ИИ",
        questions,
    )

    assert [answer.category for answer in answers] == [
        Category.SHOPPING_CLOTHES,
        Category.HOME_UTILITIES,
        Category.ENTERTAINMENT_SUBS,
    ]
    assert [answer.direction for answer in answers] == [
        Direction.EXPENSE,
        Direction.EXPENSE,
        Direction.EXPENSE,
    ]


@pytest.mark.unit
def test_build_import_questions_groups_similar_transactions() -> None:
    questions = format_import_questions(
        build_import_questions(
            [
                _tx(
                    merchant_raw="Ivan Ivanov",
                    amount="1000",
                    direction=Direction.TRANSFER,
                    category=Category.TRANSFER_INTERNAL,
                    confidence=0.6,
                ),
                _tx(
                    merchant_raw="Ivan Ivanov",
                    amount="250",
                    direction=Direction.TRANSFER,
                    category=Category.TRANSFER_INTERNAL,
                    confidence=0.6,
                ),
            ],
        )
    )

    assert questions == ["Перевод: «Ivan Ivanov», 01.04.2026, 2 операций на 1250.00 ₽. Что это?"]


@pytest.mark.unit
def test_parse_clarification_answers_ignores_unknown_labels() -> None:
    transaction = _tx(
        merchant_raw="Unknown",
        amount="100",
        direction=Direction.EXPENSE,
        category=Category.UNCLASSIFIED,
        confidence=0.0,
    )
    transaction.import_hash = "hash-1"

    answers = parse_clarification_answers(
        "1 что-то странное",
        build_import_questions([transaction]),
    )

    assert answers == []
