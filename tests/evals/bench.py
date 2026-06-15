"""Сквозной бенч latency p95 + cost per run на текущем графе (REQ-02).

Повторяет инвокацию бота 1-в-1: тот же `build_supervisor_graph(checkpointer)`,
тот же `make_callback_handler()`, тот же shape стейта и `config`. Сидлит
выделенную тестовую семью с реалистичной историей трат (3 месяца, подписки,
зарплата), чтобы coach/advisor/subscriptions реально тянули контекст из БД —
иначе cost занижен (оговорка в README про «пустой датасет»).

Latency меряем wall-clock локально (надёжнее, чем латентность из LangFuse),
cost тянем из LangFuse по уникальной сессии прогона. Первый запрос — холодный
прогрев, в p95 не включается.

    just up                # инфра должна быть поднята
    uv run python -m tests.evals.bench

Артефакт воспроизводимый — оставлен в репо как benchmark-харнесс (требование
диплома «benchmark ≥10 запросов»). Тратит бюджет OpenRouter (~13 прогонов).
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from family_finance.domain.transaction import Transaction
from family_finance.domain.types import Category, Currency, Direction, TransactionSource
from family_finance.infrastructure.mcp.client import close_finance_tools, get_finance_tools
from family_finance.infrastructure.memory import get_checkpointer
from family_finance.infrastructure.observability import flush, get_langfuse, make_callback_handler
from family_finance.infrastructure.persistence.postgres_transactions import (
    PostgresTransactionRepository,
)

# Выделенная тестовая семья — не пересекается с реальными chat-id.
BENCH_TELEGRAM_USER_ID = 990001
BENCH_TELEGRAM_CHAT_ID = 990001

# 12 запросов по core-путям (по 2 на ветку), чтобы p95 не был артефактом одной ноды.
QUERIES: list[str] = [
    "Кофе в Старбакс 350 рублей",  # ingest/categorize
    "Потратил 1200 на такси вчера",  # ingest/categorize
    "Сколько я потратил на продукты в мае?",  # ledger/aggregate
    "Сколько всего ушло на еду за последний месяц?",  # ledger
    "Как часто я заказываю доставку?",  # coach behavioral
    "На что я трачу больше всего?",  # coach
    "Уложился ли я в правило 50/30/20 в мае?",  # advisor
    "Дай совет, как сократить расходы",  # advisor
    "Какие у меня подписки?",  # subscriptions
    "Есть ли регулярные платежи, которые я мог забыть?",  # subscriptions
    "Сколько потратил на еду и на транспорт в мае?",  # multi-intent fan-out
    "Покажи траты на кафе и мои подписки",  # multi-intent fan-out
]


def _month_start(now: datetime, months_back: int) -> datetime:
    """Первое число месяца на ``months_back`` назад от ``now`` (UTC, 10:00)."""
    year, month = now.year, now.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    return datetime(year, month, 1, 10, 0, tzinfo=UTC)


def _build_history(family_id: uuid.UUID, member_id: uuid.UUID) -> list[Transaction]:
    """Реалистичная история: текущий месяц + 2 предыдущих, подписки, зарплата.

    Месяцы считаются ОТНОСИТЕЛЬНО ``now`` — иначе advisor/coach (анализируют
    ТЕКУЩИЙ месяц, см. ``analyze_spending`` → ``current_moscow_month(now)``)
    упёрлись бы в пустой период и срабатывал бы honest-гейт «нет данных» вместо
    реального ReAct-цикла (тогда cost занижен до $0, а демо выглядит сломанным).

    Детерминированный ``import_hash`` на запись → повторный прогон идемпотентен
    (``add_many`` делает ON CONFLICT DO NOTHING).
    """
    txs: list[Transaction] = []
    now = datetime.now(UTC)

    def tx(
        day: datetime,
        amount: str,
        merchant: str,
        category: Category,
        direction: Direction = Direction.EXPENSE,
    ) -> None:
        if day > now:  # не сеем «будущие» траты внутри текущего месяца
            return
        txs.append(
            Transaction(
                family_id=family_id,
                member_id=member_id,
                occurred_at=day,
                amount=Decimal(amount),
                currency=Currency.RUB,
                direction=direction,
                merchant_raw=merchant,
                merchant_normalized=merchant,
                category=category,
                confidence=0.95,
                source=TransactionSource.BANK_CSV,
                import_hash=f"bench:{merchant}:{day.date()}:{amount}",
            )
        )

    # Текущий месяц + два предыдущих (относительно сейчас).
    for months_back in (2, 1, 0):
        base = _month_start(now, months_back)
        # Зарплата раз в месяц.
        tx(base, "180000.00", "ООО Работодатель", Category.UNCLASSIFIED, Direction.INCOME)
        # Подписки (регулярные, тот же мерчант каждый месяц → detect_recurring).
        tx(base + timedelta(days=2), "599.00", "Netflix", Category.ENTERTAINMENT_SUBS)
        tx(base + timedelta(days=3), "299.00", "Яндекс Плюс", Category.ENTERTAINMENT_SUBS)
        tx(base + timedelta(days=4), "3500.00", "World Class фитнес", Category.HEALTH_FITNESS)
        tx(base + timedelta(days=5), "1200.00", "Spotify Premium", Category.ENTERTAINMENT_SUBS)
        # Продукты — еженедельно.
        for week in range(4):
            tx(
                base + timedelta(days=7 * week + 1),
                "4200.00",
                "Пятёрочка",
                Category.FOOD_GROCERIES,
            )
        # Рестораны и кофе.
        tx(base + timedelta(days=8), "2300.00", "Кафе Шоколадница", Category.FOOD_RESTAURANT)
        tx(base + timedelta(days=15), "1800.00", "Тануки", Category.FOOD_RESTAURANT)
        tx(base + timedelta(days=10), "350.00", "Старбакс", Category.FOOD_COFFEE)
        tx(base + timedelta(days=20), "420.00", "Cofix", Category.FOOD_COFFEE)
        # Доставка еды — несколько раз в месяц.
        for d in (6, 13, 19, 26):
            tx(base + timedelta(days=d), "1100.00", "Яндекс Еда", Category.FOOD_DELIVERY)
        # Транспорт.
        tx(base + timedelta(days=9), "1500.00", "Яндекс Такси", Category.TRANSPORT_TAXI)
        tx(base + timedelta(days=22), "1700.00", "Яндекс Такси", Category.TRANSPORT_TAXI)
        tx(base + timedelta(days=12), "3000.00", "Лукойл АЗС", Category.TRANSPORT_FUEL)
        # ЖКХ и связь.
        tx(base + timedelta(days=11), "6500.00", "Мосэнергосбыт ЖКХ", Category.HOME_UTILITIES)
        tx(base + timedelta(days=11), "800.00", "МТС связь", Category.HOME_TELECOM)
        # Прочее.
        tx(base + timedelta(days=17), "4500.00", "Wildberries", Category.SHOPPING_GENERIC)
        tx(base + timedelta(days=24), "1300.00", "Аптека Ригла", Category.HEALTH_PHARMACY)

    return txs


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank перцентиль (как в README-прогоне 2026-05-31)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


async def _fetch_run_cost(session_id: str, n_expected: int) -> tuple[float | None, int]:
    """Сумма totalCost трейсов прогона из LangFuse (с ретраями на ingestion-задержку)."""
    api = get_langfuse().api
    for attempt in range(12):  # до ~60 c ожидания ingestion в ClickHouse
        await asyncio.sleep(5)
        try:
            page = api.trace.list(session_id=session_id, limit=100)
        except Exception:  # observability не должна ронять бенч
            continue
        traces = list(page.data)
        if len(traces) >= n_expected:
            costs = [float(getattr(t, "total_cost", 0.0) or 0.0) for t in traces]
            print(f"  LangFuse: получено {len(traces)} трейсов (попытка {attempt + 1})")
            return sum(costs), len(traces)
    return None, 0


async def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    session_id = f"bench:{run_id}"
    print(f"=== Bench REQ-02 — run {run_id} ===\n")

    repo = PostgresTransactionRepository()
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=BENCH_TELEGRAM_USER_ID,
        name="Bench Family",
    )
    history = _build_history(family_id, member_id)
    inserted = await repo.add_many(history)
    print(
        f"Семья {family_id}: история {len(history)} транзакций "
        f"(новых вставлено: {len(inserted)}).\n"
    )

    async with get_checkpointer() as checkpointer:
        from family_finance.agents import build_supervisor_graph

        graph = build_supervisor_graph(checkpointer)
        await get_finance_tools()  # прогрев MCP-сессии, как в боте

        def invoke_config(thread_id: str) -> RunnableConfig:
            callbacks = cast("list[BaseCallbackHandler]", [make_callback_handler()])
            return {
                "configurable": {"thread_id": thread_id},
                "callbacks": callbacks,
                "metadata": {
                    "langfuse_user_id": str(BENCH_TELEGRAM_USER_ID),
                    "langfuse_session_id": session_id,
                    "langfuse_tags": ["eval", "bench", "phase3.9"],
                    "langfuse_trace_name": "bench-run",
                },
            }

        async def run_one(query: str, idx: int) -> float:
            thread_id = f"{session_id}:q{idx}"  # свежий тред на запрос = независимый прогон
            state = {
                "messages": [HumanMessage(content=query)],
                "telegram_user_id": BENCH_TELEGRAM_USER_ID,
                "telegram_chat_id": BENCH_TELEGRAM_CHAT_ID,
                "family_id": str(family_id),
                "member_id": str(member_id),
            }
            start = time.perf_counter()
            await graph.ainvoke(state, config=invoke_config(thread_id))
            return time.perf_counter() - start

        # Холодный прогрев — отдельно, в p95 не включаем.
        cold = await run_one("привет", idx=0)
        print(f"Холодный прогрев: {cold:.2f} с (исключён из p95)\n")

        latencies: list[float] = []
        for i, q in enumerate(QUERIES, start=1):
            dt = await run_one(q, idx=i)
            latencies.append(dt)
            print(f"  q{i:>2} {dt:5.2f} с  | {q}")

    flush()
    print("\nЖду ingestion в LangFuse для cost…")
    total_cost, n_traces = await _fetch_run_cost(session_id, n_expected=len(QUERIES))

    p95 = _percentile(latencies, 95)
    median = statistics.median(latencies)
    print("\n=== Результаты ===")
    print(f"Запросов (steady-state): {len(latencies)}")
    print(f"Latency p95:    {p95:.2f} с")
    print(f"Latency median: {median:.2f} с")
    print(f"Холодный старт: {cold:.2f} с (разовый, не в p95)")
    if total_cost is not None and n_traces:
        print(
            f"Cost per run (avg): ${total_cost / n_traces:.5f}  "
            f"(суммарно ${total_cost:.4f} по {n_traces} трейсам)"
        )
    else:
        print(
            "Cost: трейсы ещё не доехали в LangFuse — посмотри в UI по "
            f"session_id={session_id} (фильтр tag=bench)"
        )

    await close_finance_tools()


if __name__ == "__main__":
    asyncio.run(main())
