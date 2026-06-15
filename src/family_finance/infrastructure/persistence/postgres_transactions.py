"""Postgres adapter for transaction persistence."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime
from decimal import Decimal

import asyncpg

from family_finance.application.ports import LedgerBucket, LedgerEntry, LedgerSummary
from family_finance.domain import (
    SUBSCRIPTION_CATEGORIES,
    Budget,
    BudgetStatus,
    Category,
    Direction,
    SavingsGoal,
    Subscription,
    Transaction,
)
from family_finance.infrastructure.settings import get_settings

# ── Shared connection pool ───────────────────────────────────────────────────
# Keyed by DSN so tests can use an isolated database without breaking prod.
# Pools live for the process lifetime; aiogram + LangGraph don't have a clean
# shutdown hook for repository singletons.
_pools: dict[str, asyncpg.Pool] = {}
_pool_lock = asyncio.Lock()


async def _get_pool(dsn: str) -> asyncpg.Pool:
    """Return a process-wide pool for *dsn*, creating it on first use."""
    pool = _pools.get(dsn)
    if pool is not None:
        return pool
    async with _pool_lock:
        pool = _pools.get(dsn)
        if pool is None:
            pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
            _pools[dsn] = pool
        return pool


@contextlib.asynccontextmanager
async def loop_local_pool(dsn: str | None = None) -> AsyncIterator[asyncpg.Pool]:
    """Bind a fresh pool on the *current* event loop, bypassing the shared cache.

    For offline runners that drive each task on a throwaway loop (the LangFuse
    eval ``experiment`` — each ``run_experiment`` runs in its own thread+loop):
    the process-wide cached pool would be bound to an already-closed loop and
    raise «connection was closed» / «another operation is in progress». This
    installs a pool tied to the running loop for the block, then restores the
    cache. Safe only for serial use of *dsn* (the eval runner pins cascade to
    ``max_concurrency=1``); the long-lived bot keeps using ``_get_pool``.
    """
    resolved = dsn or get_settings().database_url.get_secret_value()
    pool = await asyncpg.create_pool(dsn=resolved, min_size=1, max_size=2, command_timeout=30)
    previous = _pools.get(resolved)
    _pools[resolved] = pool
    try:
        yield pool
    finally:
        if previous is not None:
            _pools[resolved] = previous
        else:
            _pools.pop(resolved, None)
        await pool.close()


# ── Whitelisted SQL fragments for the flexible query tools ───────────────────
# group_by / order_by come from a closed enum and map to FIXED SQL expressions
# here. Nothing user-supplied is ever interpolated — keeps "safe fixed SQL".
# Time buckets use Europe/Moscow local dates so "по дням" matches the user's
# calendar, not UTC.
_AGG_BUCKET_SQL: dict[str, str] = {
    "day": "to_char((occurred_at AT TIME ZONE 'Europe/Moscow')::date, 'YYYY-MM-DD')",
    "week": (
        "to_char(date_trunc('week', occurred_at AT TIME ZONE 'Europe/Moscow')::date, 'YYYY-MM-DD')"
    ),
    "month": "to_char((occurred_at AT TIME ZONE 'Europe/Moscow')::date, 'YYYY-MM')",
    "category": "category",
    "merchant": "COALESCE(NULLIF(merchant_raw, ''), '(без продавца)')",
    "total": "'total'",
}
# Category/merchant → biggest first; time buckets → chronological.
_AGG_ORDER_SQL: dict[str, str] = {
    "day": "bucket ASC",
    "week": "bucket ASC",
    "month": "bucket ASC",
    "category": "total_num DESC",
    "merchant": "total_num DESC",
    "total": "bucket ASC",
}
_LIST_ORDER_SQL: dict[str, str] = {
    "date_desc": "occurred_at DESC",
    "amount_desc": "amount DESC",
}


class PostgresTransactionRepository:
    """TransactionRepository backed by the domain `transaction` table."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or get_settings().database_url.get_secret_value()

    async def ensure_member_for_telegram(
        self,
        *,
        telegram_user_id: int,
        name: str,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        """Return existing member mapping or create a local default family/member."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn, conn.transaction():
            existing = await conn.fetchrow(
                """
                SELECT family_id, member_id
                FROM family_member
                WHERE telegram_user_id = $1
                """,
                telegram_user_id,
            )
            if existing is not None:
                return existing["family_id"], existing["member_id"]

            family_id = await conn.fetchval(
                "INSERT INTO family(name) VALUES($1) RETURNING family_id",
                f"{name}'s family",
            )
            member_id = await conn.fetchval(
                """
                INSERT INTO family_member(family_id, name, telegram_user_id)
                VALUES($1, $2, $3)
                RETURNING member_id
                """,
                family_id,
                name,
                telegram_user_id,
            )
            return family_id, member_id

    async def add_many(self, transactions: Sequence[Transaction]) -> list[Transaction]:
        """Insert transactions idempotently.

        Returns the list of *actually inserted* transactions (in input order).
        Duplicates (conflict on ``import_hash``) are silently skipped so the
        caller can categorise / write to memory only the new rows.
        """
        if not transactions:
            return []

        pool = await _get_pool(self._dsn)
        inserted_hashes: set[str] = set()
        async with pool.acquire() as conn, conn.transaction():
            for tx in transactions:
                row = await conn.fetchval(
                    """
                    INSERT INTO "transaction"(
                        transaction_id,
                        family_id,
                        member_id,
                        occurred_at,
                        ingested_at,
                        amount,
                        currency,
                        direction,
                        merchant_raw,
                        merchant_normalized,
                        category,
                        subcategory_freetext,
                        confidence,
                        needs_review,
                        source,
                        source_file,
                        receipt_id,
                        receipt_fns_qr,
                        tags,
                        import_hash
                    )
                    VALUES(
                        $1, $2, $3, $4, $5,
                        $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20
                    )
                    ON CONFLICT (import_hash) DO NOTHING
                    RETURNING import_hash
                    """,
                    tx.transaction_id,
                    tx.family_id,
                    tx.member_id,
                    tx.occurred_at,
                    tx.ingested_at,
                    tx.amount,
                    tx.currency.value,
                    tx.direction.value,
                    tx.merchant_raw,
                    tx.merchant_normalized,
                    tx.category.value,
                    tx.subcategory_freetext,
                    tx.confidence,
                    tx.needs_review,
                    tx.source.value,
                    tx.source_file,
                    tx.receipt_id,
                    tx.receipt_fns_qr,
                    list(tx.tags),
                    tx.import_hash,
                )
                if row is not None:
                    inserted_hashes.add(row)

        return [tx for tx in transactions if tx.import_hash in inserted_hashes]

    async def aggregate(
        self,
        *,
        family_id: uuid.UUID,
        categories: Sequence[Category],
        directions: Sequence[Direction],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> LedgerSummary:
        """Aggregate transactions with safe fixed SQL and mandatory family scope."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM(amount), 0)::TEXT AS total,
                    COUNT(*) AS count
                FROM "transaction"
                WHERE family_id = $1
                  AND category = ANY($2::TEXT[])
                  AND direction = ANY($3::TEXT[])
                  AND ($4::TIMESTAMPTZ IS NULL OR occurred_at >= $4)
                  AND ($5::TIMESTAMPTZ IS NULL OR occurred_at < $5)
                """,
                family_id,
                [category.value for category in categories],
                [direction.value for direction in directions],
                start,
                end,
            )
        if row is None:
            return LedgerSummary(total=Decimal("0"), count=0)
        return LedgerSummary(total=Decimal(row["total"]), count=row["count"])

    async def query_aggregates(
        self,
        *,
        family_id: uuid.UUID,
        group_by: str,
        then_by: str | None = None,
        categories: Sequence[Category] = (),
        directions: Sequence[Direction] = (),
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[LedgerBucket]:
        """Grouped sums over a flexible dimension. See port docstring.

        ``group_by``/``then_by`` are validated against a fixed whitelist before
        they touch SQL, so the resulting query is still fully parameterised +
        fixed-shape. Empty ``categories``/``directions`` → no filter on that
        column. When ``then_by`` is set, the query groups by two dimensions and
        each row carries a ``subbucket``.
        """
        bucket_expr = _AGG_BUCKET_SQL.get(group_by)
        order_expr = _AGG_ORDER_SQL.get(group_by)
        if bucket_expr is None or order_expr is None:
            raise ValueError(f"unsupported group_by: {group_by!r}")

        cat_filter = [c.value for c in categories] or None
        dir_filter = [d.value for d in directions] or None
        safe_limit = max(1, min(int(limit), 1000))

        # bucket_expr/order_expr are whitelisted fragments (see _AGG_*_SQL); every
        # value below is a bound parameter ($1..$6) — no user input is interpolated.
        if then_by is not None:
            return await self._query_aggregates_2d(
                family_id=family_id,
                bucket_expr=bucket_expr,
                then_by=then_by,
                cat_filter=cat_filter,
                dir_filter=dir_filter,
                start=start,
                end=end,
                safe_limit=safe_limit,
            )

        sql = f"""
            SELECT
                {bucket_expr} AS bucket,
                COALESCE(SUM(amount), 0)::TEXT AS total,
                COALESCE(SUM(amount), 0) AS total_num,
                COUNT(*) AS count
            FROM "transaction"
            WHERE family_id = $1
              AND ($2::TEXT[] IS NULL OR category = ANY($2::TEXT[]))
              AND ($3::TEXT[] IS NULL OR direction = ANY($3::TEXT[]))
              AND ($4::TIMESTAMPTZ IS NULL OR occurred_at >= $4)
              AND ($5::TIMESTAMPTZ IS NULL OR occurred_at < $5)
            GROUP BY bucket
            ORDER BY {order_expr}
            LIMIT $6
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, family_id, cat_filter, dir_filter, start, end, safe_limit)
        return [
            LedgerBucket(
                bucket=str(row["bucket"]),
                total=Decimal(row["total"]),
                count=row["count"],
            )
            for row in rows
        ]

    async def _query_aggregates_2d(
        self,
        *,
        family_id: uuid.UUID,
        bucket_expr: str,
        then_by: str,
        cat_filter: list[str] | None,
        dir_filter: list[str] | None,
        start: datetime | None,
        end: datetime | None,
        safe_limit: int,
    ) -> list[LedgerBucket]:
        """Two-dimensional grouped sums (``group_by`` + ``then_by``).

        Rows come back ordered by primary bucket ascending, then by amount
        descending inside each bucket — so «по дням по категориям» reads as
        chronological days with the biggest category first.
        """
        sub_expr = _AGG_BUCKET_SQL.get(then_by)
        if sub_expr is None:
            raise ValueError(f"unsupported then_by: {then_by!r}")

        # bucket_expr/sub_expr are whitelisted fragments; filters are bound params.
        sql = f"""
            SELECT
                {bucket_expr} AS bucket,
                {sub_expr} AS subbucket,
                COALESCE(SUM(amount), 0)::TEXT AS total,
                COUNT(*) AS count
            FROM "transaction"
            WHERE family_id = $1
              AND ($2::TEXT[] IS NULL OR category = ANY($2::TEXT[]))
              AND ($3::TEXT[] IS NULL OR direction = ANY($3::TEXT[]))
              AND ($4::TIMESTAMPTZ IS NULL OR occurred_at >= $4)
              AND ($5::TIMESTAMPTZ IS NULL OR occurred_at < $5)
            GROUP BY bucket, subbucket
            ORDER BY bucket ASC, COALESCE(SUM(amount), 0) DESC
            LIMIT $6
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, family_id, cat_filter, dir_filter, start, end, safe_limit)
        return [
            LedgerBucket(
                bucket=str(row["bucket"]),
                subbucket=str(row["subbucket"]),
                total=Decimal(row["total"]),
                count=row["count"],
            )
            for row in rows
        ]

    async def list_transactions(
        self,
        *,
        family_id: uuid.UUID,
        categories: Sequence[Category] = (),
        directions: Sequence[Direction] = (),
        start: datetime | None = None,
        end: datetime | None = None,
        order_by: str = "date_desc",
        limit: int = 20,
    ) -> list[LedgerEntry]:
        """Raw transaction rows, newest-first or biggest-first (``order_by``)."""
        order_expr = _LIST_ORDER_SQL.get(order_by)
        if order_expr is None:
            raise ValueError(f"unsupported order_by: {order_by!r}")

        cat_filter = [c.value for c in categories] or None
        dir_filter = [d.value for d in directions] or None
        safe_limit = max(1, min(int(limit), 200))

        # order_expr is a whitelisted fragment (see _LIST_ORDER_SQL); all filter
        # values are bound parameters ($1..$6) — nothing user-supplied interpolated.
        sql = f"""
            SELECT
                occurred_at,
                amount::TEXT AS amount,
                direction,
                category,
                COALESCE(merchant_raw, '') AS merchant
            FROM "transaction"
            WHERE family_id = $1
              AND ($2::TEXT[] IS NULL OR category = ANY($2::TEXT[]))
              AND ($3::TEXT[] IS NULL OR direction = ANY($3::TEXT[]))
              AND ($4::TIMESTAMPTZ IS NULL OR occurred_at >= $4)
              AND ($5::TIMESTAMPTZ IS NULL OR occurred_at < $5)
            ORDER BY {order_expr}
            LIMIT $6
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, family_id, cat_filter, dir_filter, start, end, safe_limit)
        return [
            LedgerEntry(
                occurred_at=row["occurred_at"],
                amount=Decimal(row["amount"]),
                direction=Direction(row["direction"]),
                category=Category(row["category"]),
                merchant=row["merchant"],
            )
            for row in rows
        ]

    async def classify_by_import_hashes(
        self,
        *,
        family_id: uuid.UUID,
        import_hashes: Sequence[str],
        category: Category,
        direction: Direction,
        subcategory_freetext: str | None = None,
        confidence: float = 1.0,
        needs_review: bool | None = None,
    ) -> int:
        """Apply a classification (LLM or user-confirmed) to imported transactions.

        ``needs_review`` defaults to ``confidence < 0.7`` so the categorizer
        can omit it; the clarify node passes ``False`` explicitly after the
        user picks a category.
        """
        if not import_hashes:
            return 0

        effective_needs_review = confidence < 0.7 if needs_review is None else needs_review

        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE "transaction"
                SET category = $3,
                    direction = $4,
                    subcategory_freetext = $5,
                    confidence = $6,
                    needs_review = $7
                WHERE family_id = $1
                  AND import_hash = ANY($2::TEXT[])
                """,
                family_id,
                list(import_hashes),
                category.value,
                direction.value,
                subcategory_freetext,
                confidence,
                effective_needs_review,
            )
        return int(result.rsplit(" ", maxsplit=1)[-1])

    async def set_budget(
        self,
        *,
        family_id: uuid.UUID,
        category: Category,
        monthly_limit: Decimal,
    ) -> None:
        """Insert or replace a monthly budget for one (family, category) pair."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO budget (family_id, category, monthly_limit)
                VALUES ($1, $2, $3)
                ON CONFLICT (family_id, category) DO UPDATE
                SET monthly_limit = EXCLUDED.monthly_limit,
                    updated_at = NOW()
                """,
                family_id,
                category.value,
                monthly_limit,
            )

    async def delete_budget(
        self,
        *,
        family_id: uuid.UUID,
        category: Category,
    ) -> bool:
        """Remove a budget. Returns True if a row was actually deleted."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM budget
                WHERE family_id = $1 AND category = $2
                """,
                family_id,
                category.value,
            )
        return int(result.rsplit(" ", maxsplit=1)[-1]) > 0

    async def get_budgets(self, *, family_id: uuid.UUID) -> list[Budget]:
        """Return all budgets configured for a family."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT family_id, category, monthly_limit
                FROM budget
                WHERE family_id = $1
                ORDER BY category
                """,
                family_id,
            )
        return [
            Budget(
                family_id=row["family_id"],
                category=Category(row["category"]),
                monthly_limit=Decimal(row["monthly_limit"]),
            )
            for row in rows
        ]

    async def get_budget_status(
        self,
        *,
        family_id: uuid.UUID,
        month_start: datetime,
        month_end: datetime,
    ) -> list[BudgetStatus]:
        """Return BudgetStatus for every budget, summing this month's spend.

        The caller supplies the ``[month_start, month_end)`` bracket — the
        budgets agent computes it from the current Europe/Moscow month so
        the boundary logic isn't duplicated in SQL.
        """
        budgets = await self.get_budgets(family_id=family_id)
        if not budgets:
            return []

        breakdown = await self.category_breakdown(
            family_id=family_id,
            start=month_start,
            end=month_end,
        )
        spent_by_category: dict[Category, Decimal] = {cat: total for cat, total, _ in breakdown}
        return [
            BudgetStatus(
                budget=b,
                spent_this_month=spent_by_category.get(b.category, Decimal("0")),
            )
            for b in budgets
        ]

    async def set_savings_goal(
        self,
        *,
        family_id: uuid.UUID,
        target_amount: Decimal,
        target_date: date | None = None,
    ) -> None:
        """Insert or replace the family's single savings goal.

        Replacing the goal resets ``created_at`` so progress is measured from
        the new starting line.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO savings_goal (family_id, target_amount, target_date)
                VALUES ($1, $2, $3)
                ON CONFLICT (family_id) DO UPDATE
                SET target_amount = EXCLUDED.target_amount,
                    target_date = EXCLUDED.target_date,
                    created_at = NOW(),
                    updated_at = NOW()
                """,
                family_id,
                target_amount,
                target_date,
            )

    async def get_savings_goal(self, *, family_id: uuid.UUID) -> SavingsGoal | None:
        """Return the family's savings goal, or None if none is set."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT family_id, target_amount, target_date, created_at
                FROM savings_goal
                WHERE family_id = $1
                """,
                family_id,
            )
        if row is None:
            return None
        return SavingsGoal(
            family_id=row["family_id"],
            target_amount=Decimal(row["target_amount"]),
            target_date=row["target_date"],
            created_at=row["created_at"],
        )

    async def delete_savings_goal(self, *, family_id: uuid.UUID) -> bool:
        """Remove the family's savings goal. True if a row was deleted."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM savings_goal WHERE family_id = $1",
                family_id,
            )
        return int(result.rsplit(" ", maxsplit=1)[-1]) > 0

    async def net_cashflow(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> Decimal:
        """Net savings over ``[start, end)``: income + refunds − expenses.

        Internal transfers are ignored — moving money between own wallets is
        neither earning nor spending.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            net = await conn.fetchval(
                """
                SELECT COALESCE(SUM(
                    CASE direction
                        WHEN 'income' THEN amount
                        WHEN 'refund' THEN amount
                        WHEN 'expense' THEN -amount
                        ELSE 0
                    END
                ), 0)::TEXT
                FROM "transaction"
                WHERE family_id = $1
                  AND occurred_at >= $2
                  AND occurred_at < $3
                """,
                family_id,
                start,
                end,
            )
        return Decimal(net)

    async def set_digest_cron(
        self,
        *,
        member_id: uuid.UUID,
        cron: str | None,
    ) -> None:
        """Persist (or clear) the member's preferred digest schedule."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE family_member
                SET digest_cron = $2
                WHERE member_id = $1
                """,
                member_id,
                cron,
            )

    async def iter_digest_schedules(
        self,
    ) -> list[tuple[uuid.UUID, uuid.UUID, int, str]]:
        """Return active digest schedules across all members.

        One row per member with a non-null ``digest_cron`` AND a linked
        ``telegram_user_id`` (we can only deliver to chats we know about).
        Tuple shape: ``(family_id, member_id, telegram_user_id, cron)``.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT family_id, member_id, telegram_user_id, digest_cron
                FROM family_member
                WHERE digest_cron IS NOT NULL
                  AND telegram_user_id IS NOT NULL
                """,
            )
        return [
            (
                row["family_id"],
                row["member_id"],
                row["telegram_user_id"],
                row["digest_cron"],
            )
            for row in rows
        ]

    async def iter_telegram_families(self) -> list[tuple[uuid.UUID, int]]:
        """Return (family_id, telegram_user_id) pairs for every linked family.

        Used by the scheduler to know which chats to push the weekly digest
        into. In private Telegram chats ``chat_id == user_id``, which is the
        case we currently support.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (family_id)
                    family_id,
                    telegram_user_id
                FROM family_member
                WHERE telegram_user_id IS NOT NULL
                ORDER BY family_id, created_at ASC
                """,
            )
        return [(row["family_id"], row["telegram_user_id"]) for row in rows]

    async def category_breakdown(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
        direction: Direction = Direction.EXPENSE,
    ) -> list[tuple[Category, Decimal, int]]:
        """Return per-category totals for ``[start, end)``, biggest first.

        Used by the weekly digest. One SQL aggregate, no LLM.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    category,
                    SUM(amount)::TEXT AS total,
                    COUNT(*) AS n
                FROM "transaction"
                WHERE family_id = $1
                  AND direction = $2
                  AND occurred_at >= $3
                  AND occurred_at < $4
                GROUP BY category
                ORDER BY SUM(amount) DESC
                """,
                family_id,
                direction.value,
                start,
                end,
            )
        return [(Category(row["category"]), Decimal(row["total"]), row["n"]) for row in rows]

    async def detect_recurring(
        self,
        *,
        family_id: uuid.UUID,
        min_occurrences: int = 3,
        lookback_days: int = 365,
        min_cadence_days: int = 20,
        max_cadence_days: int = 45,
    ) -> list[Subscription]:
        """Detect recurring expenses (subscriptions, regular bills) for a family.

        Heuristic — pure SQL aggregate, no LLM:
          * group by ``merchant_raw`` (raw normalized form)
          * count >= ``min_occurrences`` in the lookback window
          * average inter-payment gap within ``[min_cadence_days, max_cadence_days]``
            (default 20..45 days ≈ monthly)
          * direction = EXPENSE only
          * category in ``SUBSCRIPTION_CATEGORIES`` — иначе еженедельная продуктовая
            корзина (Пятёрочка) или аптека ловятся как «подписка» (QA-03)

        Returns a list of :class:`Subscription` value objects, newest-seen first.
        """
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH gaps AS (
                    SELECT
                        merchant_raw,
                        category,
                        amount,
                        occurred_at,
                        LAG(occurred_at) OVER (
                            PARTITION BY merchant_raw
                            ORDER BY occurred_at
                        ) AS prev_at
                    FROM "transaction"
                    WHERE family_id = $1
                      AND direction = 'expense'
                      AND occurred_at >= NOW() - make_interval(days => $3)
                      AND category = ANY($6::text[])
                ),
                summary AS (
                    SELECT
                        merchant_raw,
                        (array_agg(category ORDER BY occurred_at DESC))[1] AS category,
                        COUNT(*) AS occurrences,
                        AVG(amount)::TEXT AS avg_amount,
                        (array_agg(amount ORDER BY occurred_at DESC))[1]::TEXT AS last_amount,
                        MAX(occurred_at) AS last_seen,
                        AVG(
                            EXTRACT(EPOCH FROM (occurred_at - prev_at)) / 86400.0
                        ) FILTER (WHERE prev_at IS NOT NULL) AS avg_cadence_days
                    FROM gaps
                    GROUP BY merchant_raw
                )
                SELECT *
                FROM summary
                WHERE occurrences >= $2
                  AND avg_cadence_days BETWEEN $4 AND $5
                ORDER BY last_seen DESC
                """,
                family_id,
                min_occurrences,
                lookback_days,
                min_cadence_days,
                max_cadence_days,
                [c.value for c in SUBSCRIPTION_CATEGORIES],
            )

        return [
            Subscription(
                merchant=row["merchant_raw"],
                category=Category(row["category"]),
                cadence_days=round(row["avg_cadence_days"]),
                average_amount=Decimal(row["avg_amount"]),
                last_amount=Decimal(row["last_amount"]),
                last_seen=row["last_seen"],
                occurrences=row["occurrences"],
            )
            for row in rows
        ]
