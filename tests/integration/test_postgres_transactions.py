"""Integration tests for Postgres transaction repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import asyncpg
import pytest

from family_finance.domain import Category, Currency, Direction, Transaction, TransactionSource
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import get_settings


async def _skip_if_postgres_unavailable() -> None:
    try:
        conn = await asyncpg.connect(dsn=get_settings().database_url.get_secret_value())
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres is unavailable: {exc}")
    else:
        await conn.close()


@pytest.mark.integration
async def test_aggregate_filters_by_family_category_direction_and_period() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"Integration {telegram_user_id}",
    )

    transactions = [
        Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=datetime(2026, 4, 10, tzinfo=UTC),
            amount=Decimal("100.00"),
            currency=Currency.RUB,
            direction=Direction.EXPENSE,
            merchant_raw="Integration pharmacy",
            category=Category.HEALTH_PHARMACY,
            confidence=1.0,
            source=TransactionSource.BANK_CSV,
            import_hash=f"integration:{uuid.uuid4()}",
        ),
        Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=datetime(2026, 4, 11, tzinfo=UTC),
            amount=Decimal("50.00"),
            currency=Currency.RUB,
            direction=Direction.EXPENSE,
            merchant_raw="Integration grocery",
            category=Category.FOOD_GROCERIES,
            confidence=1.0,
            source=TransactionSource.BANK_CSV,
            import_hash=f"integration:{uuid.uuid4()}",
        ),
        Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
            amount=Decimal("30.00"),
            currency=Currency.RUB,
            direction=Direction.EXPENSE,
            merchant_raw="Integration pharmacy may",
            category=Category.HEALTH_PHARMACY,
            confidence=1.0,
            source=TransactionSource.BANK_CSV,
            import_hash=f"integration:{uuid.uuid4()}",
        ),
    ]

    assert len(await repo.add_many(transactions)) == 3

    summary = await repo.aggregate(
        family_id=family_id,
        categories=(Category.HEALTH_PHARMACY,),
        directions=(Direction.EXPENSE,),
        start=datetime(2026, 4, 1, tzinfo=UTC),
        end=datetime(2026, 5, 1, tzinfo=UTC),
    )

    assert summary.total == Decimal("100.00")
    assert summary.count == 1


@pytest.mark.integration
async def test_classify_by_import_hashes_updates_transactions() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"Integration {telegram_user_id}",
    )
    import_hash = f"integration:{uuid.uuid4()}"
    transaction = Transaction(
        family_id=family_id,
        member_id=member_id,
        occurred_at=datetime(2026, 4, 10, tzinfo=UTC),
        amount=Decimal("100.00"),
        currency=Currency.RUB,
        direction=Direction.TRANSFER,
        merchant_raw="Integration transfer",
        category=Category.TRANSFER_INTERNAL,
        confidence=0.6,
        source=TransactionSource.BANK_CSV,
        import_hash=import_hash,
    )

    assert len(await repo.add_many([transaction])) == 1
    assert (
        await repo.classify_by_import_hashes(
            family_id=family_id,
            import_hashes=(import_hash,),
            category=Category.FOOD_GROCERIES,
            direction=Direction.EXPENSE,
            subcategory_freetext="еда",
        )
        == 1
    )

    conn = await asyncpg.connect(dsn=get_settings().database_url.get_secret_value())
    try:
        row = await conn.fetchrow(
            """
            SELECT category, direction, subcategory_freetext, confidence, needs_review
            FROM "transaction"
            WHERE import_hash = $1
            """,
            import_hash,
        )
    finally:
        await conn.close()

    assert row is not None
    assert row["category"] == Category.FOOD_GROCERIES.value
    assert row["direction"] == Direction.EXPENSE.value
    assert row["subcategory_freetext"] == "еда"
    assert row["confidence"] == 1.0
    assert row["needs_review"] is False
