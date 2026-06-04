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

    async def iter_telegram_families(self) -> list[tuple[uuid.UUID, int]]: ...

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
