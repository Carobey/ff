"""
Application ports — типизированные интерфейсы для infrastructure-адаптеров.

Используется typing.Protocol (структурный subtyping) — идиоматично для 2026.
Любой класс с совпадающими методами автоматически считается реализацией.

Это позволяет:
- Тестировать use cases с моками без наследования
- Свободно менять адаптеры (Postgres → SQLite → in-memory) без рефакторинга
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from family_finance.domain import (
    Budget,
    BudgetStatus,
    Category,
    Direction,
    SavingsGoal,
    Subscription,
    Transaction,
)


@dataclass(frozen=True)
class LedgerSummary:
    """Aggregate amount/count for a ledger query."""

    total: Decimal
    count: int


@dataclass(frozen=True)
class LedgerBucket:
    """One row of a grouped aggregation (``query_aggregates``).

    ``bucket`` is the group key as a display string — a date (``2026-04-01``),
    a month (``2026-04``), a category value or a merchant name, depending on the
    requested ``group_by``.

    ``subbucket`` is the secondary group key when a 2-D breakdown is requested
    (``then_by``), e.g. category within a day for «по дням по категориям».
    ``None`` for a plain one-dimensional aggregation.
    """

    bucket: str
    total: Decimal
    count: int
    subbucket: str | None = None


@dataclass(frozen=True)
class LedgerEntry:
    """A single transaction row returned by ``list_transactions``."""

    occurred_at: datetime
    amount: Decimal
    direction: Direction
    category: Category
    merchant: str


@dataclass(frozen=True)
class MerchantRuleHit:
    """A matched merchant→category rule from the categorization cascade."""

    category: Category
    score: float  # fuzzy similarity 0..1
    source: str  # seed | user | llm


@runtime_checkable
class CategoryCatalog(Protocol):
    """Read-only справочник категорий (таксономия для промпта категоризатора)."""

    async def render_taxonomy(self) -> str:
        """Вернуть готовый блок «КАТЕГОРИИ» для system-промпта (по строке на код)."""
        ...


@runtime_checkable
class MerchantRuleRepository(Protocol):
    """Каскад «продавец → категория»: fuzzy-lookup + дозапись выученных правил."""

    async def lookup_many(
        self,
        *,
        family_id: uuid.UUID,
        merchants: Sequence[str],
        threshold: float,
    ) -> dict[str, MerchantRuleHit]:
        """Сопоставить продавцов правилам. Ключ результата — исходный `merchant_raw`.

        Возвращает только попадания со score ≥ threshold (промахи отсутствуют в dict).
        """
        ...

    async def upsert(
        self,
        *,
        family_id: uuid.UUID,
        merchant_raw: str,
        category: Category,
        source: str = "user",
    ) -> None:
        """Записать/обновить правило семьи (learning loop из clarify-ноды)."""
        ...


@runtime_checkable
class TransactionRepository(Protocol):
    """Persistence для транзакций."""

    async def add_many(self, transactions: Sequence[Transaction]) -> list[Transaction]:
        """
        Вставка batch'ем. Возвращает СПИСОК реально вставленных транзакций
        (в порядке входа). Дубли по `import_hash` молча отбрасываются,
        чтобы вызывающий не тратил LLM/Graphiti на уже известные строки.
        """
        ...

    async def ensure_member_for_telegram(
        self,
        *,
        telegram_user_id: int,
        name: str,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        """Вернуть (family_id, member_id) для Telegram-пользователя.

        Если маппинга нет — создать локальную семью и участника по умолчанию.
        """
        ...

    async def aggregate(
        self,
        *,
        family_id: uuid.UUID,
        categories: Sequence[Category],
        directions: Sequence[Direction],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> LedgerSummary: ...

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
        merchant_query: str | None = None,
    ) -> list[LedgerBucket]:
        """Grouped sums over a flexible dimension (day/week/month/category/merchant/total).

        Empty ``categories``/``directions`` mean "no filter". ``group_by`` must be
        one of the supported dimensions — anything else raises ``ValueError``.

        ``then_by`` adds a second grouping dimension (same whitelist) so the rows
        carry a ``subbucket``, e.g. ``group_by="day", then_by="category"`` for a
        «по дням по категориям» breakdown.
        """
        ...

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
        merchant: str | None = None,
        merchant_query: str | None = None,
    ) -> list[LedgerEntry]:
        """Raw transaction rows, newest-first or biggest-first (``order_by``).

        ``merchant`` is an exact match on the merchant bucket; ``merchant_query``
        is a normalized substring filter for free-text «расходы на <продавец>».
        """
        ...

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
    ) -> int: ...

    async def set_digest_cron(
        self,
        *,
        member_id: uuid.UUID,
        cron: str | None,
    ) -> None: ...

    async def iter_digest_schedules(
        self,
    ) -> list[tuple[uuid.UUID, uuid.UUID, int, str]]: ...

    async def set_budget(
        self,
        *,
        family_id: uuid.UUID,
        category: Category,
        monthly_limit: Decimal,
    ) -> None: ...

    async def delete_budget(
        self,
        *,
        family_id: uuid.UUID,
        category: Category,
    ) -> bool: ...

    async def get_budgets(self, *, family_id: uuid.UUID) -> list[Budget]: ...

    async def get_budget_status(
        self,
        *,
        family_id: uuid.UUID,
        month_start: datetime,
        month_end: datetime,
    ) -> list[BudgetStatus]: ...

    async def set_savings_goal(
        self,
        *,
        family_id: uuid.UUID,
        target_amount: Decimal,
        target_date: date | None = None,
    ) -> None:
        """Создать/заменить единственную цель накопления семьи.

        Замена цели сбрасывает ``created_at`` — прогресс считается заново.
        """
        ...

    async def get_savings_goal(self, *, family_id: uuid.UUID) -> SavingsGoal | None: ...

    async def delete_savings_goal(self, *, family_id: uuid.UUID) -> bool: ...

    async def net_cashflow(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> Decimal:
        """Чистый поток за ``[start, end)``: доходы + возвраты − расходы.

        Внутренние переводы игнорируются.
        """
        ...

    async def category_breakdown(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
        direction: Direction = Direction.EXPENSE,
    ) -> list[tuple[Category, Decimal, int]]: ...

    async def detect_recurring(
        self,
        *,
        family_id: uuid.UUID,
        min_occurrences: int = 3,
        lookback_days: int = 365,
        min_cadence_days: int = 20,
        max_cadence_days: int = 45,
    ) -> list[Subscription]: ...


@runtime_checkable
class BankStatementParser(Protocol):
    """Парсер банковской выписки в доменные Transaction."""

    def parse(
        self,
        content: bytes,
        *,
        family_id: uuid.UUID,
        member_id: uuid.UUID,
        source_file: str | None = None,
    ) -> list[Transaction]: ...
