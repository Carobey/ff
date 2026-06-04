"""CategorizerAgent: LLM enrichment for UNCLASSIFIED/needs_review transactions.

Запускается ПОСЛЕ ingest_node (всегда).
Обогащает UNCLASSIFIED транзакции через LLM, затем формирует список вопросов
для транзакций с низким confidence (needs_review).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, cast

import structlog
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from family_finance.agents.budgets import detect_budget_alerts
from family_finance.agents.clarifications import (
    ClarificationQuestion,
    build_import_questions,
)
from family_finance.agents.state import FinanceState
from family_finance.agents.subscriptions import detect_alerts
from family_finance.domain import Category, Direction, Transaction
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import get_settings

logger = structlog.get_logger()

# ──────────────────────────────────────────────
# Structured output schema (P1-11)
# ──────────────────────────────────────────────


class CategoryPrediction(BaseModel):
    """LLM structured output для категоризации одной транзакции."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Объяснение на русском, одно предложение")


# ──────────────────────────────────────────────
# Системный промпт (P1-12)
# ──────────────────────────────────────────────

CATEGORIZER_SYSTEM = """\
Ты — финансовый помощник семьи. Определяй категорию банковской операции.

ПРАВИЛА:
- confidence 0.9+ только при полной уверенности (известный магазин, очевидный тип)
- confidence 0.5-0.7 если название продавца непонятно или может быть несколько категорий
- confidence < 0.5 если невозможно определить → используй unclassified
- Для переводов между своими картами — всегда transfer.internal
- reasoning — по-русски, одно предложение

КАТЕГОРИИ:
food.groceries        — супермаркеты (Пятёрочка, Магнит, ВкусВилл, Перекрёсток)
food.restaurant       — рестораны, кафе, столовые, фастфуд
food.delivery         — доставка еды (Самокат, Яндекс.Еда, СберМаркет, Delivery)
transport.fuel        — АЗС (Лукойл, Роснефть, Газпром нефть)
transport.taxi        — Яндекс.Такси, Ситимобил, DiDi
transport.public      — метро, автобус, Аэроэкспресс, электричка
transport.carparts    — запчасти, шиномонтаж, автосервис
kids.clothes          — детская одежда и обувь (Детский мир)
kids.toys             — игрушки, конструкторы
kids.school           — школьные принадлежности, учебники, канцелярия
kids.activities       — секции, кружки, репетиторы, развивающие курсы
shopping.clothes      — одежда и обувь для взрослых (Wildberries, ZARA, H&M, Lamoda)
shopping.generic      — прочие покупки (Ozon, AliExpress, маркетплейсы)
home.utilities        — ЖКХ, коммуналка, интернет, мобильная связь
home.furniture        — мебель (IKEA, Hoff, Lazurit)
home.repair           — ремонт, стройматериалы (Леруа Мерлен, OBI, СТД Петрович)
home.household        — бытовая химия, хозтовары, уборка
health.pharmacy       — аптеки (36.6, АСНА, Ригла)
health.generic        — врачи, клиники, лаборатории (Инвитро, Гемотест)
entertainment.subscriptions — Яндекс.Плюс, Netflix, Spotify, ChatGPT Plus, подписки
entertainment.events  — кино, театр, концерты, экскурсии
entertainment.hobbies — спорттовары, хобби, книги
pets                  — ветеринария, зоотовары (Зоомагазин, ВетМир)
tax_ded.medical       — платная медицина с правом налогового вычета (ст.219 НК РФ)
tax_ded.education     — платное образование с правом вычета (ст.219 НК РФ)
income.salary         — зарплата, аванс
income.other          — кэшбек, возвраты, прочие доходы
transfer.internal     — перевод между своими картами/счетами
unclassified          — категорию определить невозможно
"""


# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────


async def categorizer_node(state: FinanceState) -> dict[str, Any]:
    """
    LLM-обогащение транзакций с UNCLASSIFIED/needs_review категорией.

    llm_categorize_all=False (default): только UNCLASSIFIED + needs_review.
    llm_categorize_all=True:            все транзакции кроме TRANSFER.
    """
    settings = get_settings()
    family_id = uuid.UUID(state["family_id"])
    all_transactions: list[Transaction] = list(state.get("parsed_transactions") or [])

    to_classify = _select_for_llm(all_transactions, all_=settings.llm_categorize_all)

    enriched_map: dict[str, Transaction] = {}
    if to_classify:
        enriched_map = await _run_llm_batch(to_classify, family_id=family_id)

    # Merge enriched into full list — single source of truth for both the
    # state checkpoint AND the clarification builder, so they can't disagree.
    merged = _merge_enriched(all_transactions, enriched_map)

    # Rebuild clarification questions from post-LLM state
    open_questions: list[ClarificationQuestion] = build_import_questions(merged)

    # Detect post-import alerts AFTER all classifications are persisted.
    # Both cheap: one SQL aggregate each, no LLM.
    alerts: list[str] = []
    try:
        alerts.extend(await detect_alerts(family_id=family_id))
    except Exception:
        logger.exception("subscription_alerts_failed")
    try:
        alerts.extend(await detect_budget_alerts(family_id))
    except Exception:
        logger.exception("budget_alerts_failed")

    # Build reply message
    parts: list[str] = []
    if to_classify:
        classified_count = sum(
            1 for tx in enriched_map.values() if tx.category != Category.UNCLASSIFIED
        )
        parts.append(f"LLM категоризировал {classified_count}/{len(to_classify)} транзакций.")
    if alerts:
        parts.append("\n".join(alerts))
    if open_questions:
        parts.append(
            f"Требуют уточнения {len(open_questions)} групп операций — отвечу вопросами ниже."
        )

    result: dict[str, Any] = {
        "parsed_transactions": merged,
        "open_questions": open_questions,
        "current_intent": "idle",
    }
    if parts:
        result["messages"] = [AIMessage(content=" ".join(parts))]
    return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _select_for_llm(
    transactions: list[Transaction],
    *,
    all_: bool,
) -> list[Transaction]:
    if all_:
        return [tx for tx in transactions if tx.direction != Direction.TRANSFER]
    return [tx for tx in transactions if tx.category == Category.UNCLASSIFIED or tx.needs_review]


_CATEGORIZER_CONCURRENCY = 10  # max parallel LLM calls; keeps OpenRouter happy


async def _categorize_one(
    tx: Transaction,
    model: Runnable[LanguageModelInput, CategoryPrediction],
    sem: asyncio.Semaphore,
) -> Transaction:
    """Categorize a single transaction under a concurrency semaphore."""
    async with sem:
        try:
            prediction = await model.ainvoke(
                [
                    SystemMessage(content=CATEGORIZER_SYSTEM),
                    HumanMessage(content=f"Продавец: {tx.merchant_raw}\nСумма: {tx.amount} ₽"),
                ]
            )
            return tx.model_copy(
                update={
                    "category": prediction.category,
                    "confidence": prediction.confidence,
                    "needs_review": prediction.confidence < 0.7,
                }
            )
        except Exception:
            logger.exception("categorizer_llm_failed", merchant=tx.merchant_raw)
            return tx


async def _run_llm_batch(
    transactions: list[Transaction],
    *,
    family_id: uuid.UUID,
) -> dict[str, Transaction]:
    """Run LLM on all transactions concurrently (bounded semaphore); return enriched map."""
    model = cast(
        "Runnable[LanguageModelInput, CategoryPrediction]",
        get_chat_model(tier="worker").with_structured_output(CategoryPrediction),
    )
    repository = PostgresTransactionRepository()
    sem = asyncio.Semaphore(_CATEGORIZER_CONCURRENCY)

    enriched = await asyncio.gather(*(_categorize_one(tx, model, sem) for tx in transactions))

    # Build result map keyed by import_hash | transaction_id.
    result: dict[str, Transaction] = {
        (tx.import_hash or str(tx.transaction_id)): tx for tx in enriched
    }

    # Persist classifications to DB in BATCHES grouped by (category, direction,
    # confidence). For a CSV with 100 rows that's usually 5–10 UPDATEs instead
    # of 100 — DB stays the single source of truth.
    groups: dict[tuple[Category, Direction, float], list[str]] = {}
    for tx in enriched:
        if not tx.import_hash:
            continue
        key = (tx.category, tx.direction, tx.confidence)
        groups.setdefault(key, []).append(tx.import_hash)

    for (category, direction, confidence), hashes in groups.items():
        await repository.classify_by_import_hashes(
            family_id=family_id,
            import_hashes=hashes,
            category=category,
            direction=direction,
            confidence=confidence,
        )

    return result


def _merge_enriched(
    original: list[Transaction],
    enriched: dict[str, Transaction],
) -> list[Transaction]:
    """Replace original transactions with enriched versions where available."""
    if not enriched:
        return original
    merged = []
    for tx in original:
        key = tx.import_hash or str(tx.transaction_id)
        merged.append(enriched.get(key, tx))
    return merged
