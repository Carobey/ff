"""Integration test for the recurring-payment detector against live Postgres."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
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


def _make_tx(
    *,
    family_id: uuid.UUID,
    member_id: uuid.UUID,
    merchant: str,
    amount: str,
    occurred_at: datetime,
    suffix: str,
) -> Transaction:
    return Transaction(
        family_id=family_id,
        member_id=member_id,
        occurred_at=occurred_at,
        amount=Decimal(amount),
        currency=Currency.RUB,
        direction=Direction.EXPENSE,
        merchant_raw=merchant,
        category=Category.ENTERTAINMENT_SUBS,
        confidence=1.0,
        source=TransactionSource.BANK_CSV,
        import_hash=f"integration-rec:{suffix}:{uuid.uuid4()}",
    )


@pytest.mark.integration
async def test_detect_recurring_finds_monthly_pattern() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"Recurring {telegram_user_id}",
    )

    merchant = f"NetflixTest-{uuid.uuid4().hex[:6]}"
    now = datetime.now(UTC)
    # 5 monthly payments: 4 older at 799, the most recent one bumped to 999.
    # i=0 is "today", i=4 is the oldest ~120 days ago.
    txs = [
        _make_tx(
            family_id=family_id,
            member_id=member_id,
            merchant=merchant,
            amount="999" if i == 0 else "799",
            occurred_at=now - timedelta(days=30 * i + 1),
            suffix=f"sub-{i}",
        )
        for i in range(5)
    ]
    await repo.add_many(txs)

    subs = await repo.detect_recurring(family_id=family_id)
    matching = [s for s in subs if s.merchant == merchant]

    assert len(matching) == 1
    sub = matching[0]
    assert sub.occurrences == 5
    assert 27 <= sub.cadence_days <= 33  # ≈ monthly
    assert sub.last_amount == Decimal("999.00")
    assert sub.average_amount == Decimal("839.00")  # (4×799 + 999) / 5


@pytest.mark.integration
async def test_detect_recurring_skips_one_off_purchases() -> None:
    await _skip_if_postgres_unavailable()

    repo = PostgresTransactionRepository()
    telegram_user_id = uuid.uuid4().int % 2_000_000_000
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=telegram_user_id,
        name=f"OneOff {telegram_user_id}",
    )

    merchant = f"OneOffShop-{uuid.uuid4().hex[:6]}"
    now = datetime.now(UTC)
    # Only 2 transactions — below min_occurrences=3
    txs = [
        _make_tx(
            family_id=family_id,
            member_id=member_id,
            merchant=merchant,
            amount="500",
            occurred_at=now - timedelta(days=30 * i + 1),
            suffix=f"oneoff-{i}",
        )
        for i in range(2)
    ]
    await repo.add_many(txs)

    subs = await repo.detect_recurring(family_id=family_id)
    assert all(s.merchant != merchant for s in subs)
