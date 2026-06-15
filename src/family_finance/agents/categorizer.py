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
from family_finance.application.ports import MerchantRuleHit, MerchantRuleRepository
from family_finance.domain import (
    Category,
    Direction,
    Transaction,
    direction_for_category,
)
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.memory.episode_formatter import (
    import_to_episode_body,
    make_import_episode_name,
)
from family_finance.infrastructure.memory.graphiti_client import add_episode
from family_finance.infrastructure.observability.langfuse_setup import emit_score
from family_finance.infrastructure.persistence import (
    PostgresCategoryCatalog,
    PostgresMerchantRuleRepository,
    PostgresTransactionRepository,
)
from family_finance.infrastructure.settings import get_settings

logger = structlog.get_logger()

# Strong refs to fire-and-forget Graphiti tasks so the GC can't collect them mid-flight.
_bg_tasks: set[asyncio.Task[None]] = set()

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

# Шапка промпта. Блок «КАТЕГОРИИ» рендерится из справочника (таблица `category`),
# а не хардкодится — таксономия расширяется через БД без правки кода.
CATEGORIZER_SYSTEM_HEADER = """\
Ты — финансовый помощник семьи. Определяй категорию банковской операции.

ПРАВИЛА:
- confidence 0.9+ только при полной уверенности (известный магазин, очевидный тип)
- confidence 0.7-0.9 если категория вероятна, но название не на 100% однозначно
- confidence < 0.7 если продавец непонятен или подходит несколько категорий —
  такие операции уйдут на ручную проверку (needs_review)
- confidence < 0.5 если невозможно определить → используй unclassified
- Для переводов между своими картами — всегда transfer.internal
- reasoning — по-русски, одно предложение

КАТЕГОРИИ:
"""

# Fallback на случай, если справочник в БД недоступен (БД пустая / ошибка).
# Содержит только коды enum — LLM сможет хотя бы выбрать валидную категорию.
_TAXONOMY_FALLBACK = "\n".join(c.value for c in Category)


def build_system_prompt(taxonomy: str) -> str:
    """Собрать system-промпт: статичная шапка + таксономия из справочника."""
    return CATEGORIZER_SYSTEM_HEADER + (taxonomy or _TAXONOMY_FALLBACK)


# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────


async def categorizer_node(state: FinanceState) -> dict[str, Any]:
    """
    Каскадная категоризация: справочник-правила → LLM → (далее) уточнения.

    Шаг 1 — fuzzy-поиск продавца в правилах БД: попадание ⇒ категория без LLM.
    Шаг 2 — оставшиеся (промах по правилам) уходят в LLM с таксономией из справочника.

    llm_categorize_all=False (default): обрабатываем только UNCLASSIFIED + needs_review.
    llm_categorize_all=True:            все транзакции кроме TRANSFER.
    """
    settings = get_settings()
    family_id = uuid.UUID(state["family_id"])
    all_transactions: list[Transaction] = list(state.get("parsed_transactions") or [])

    to_classify = _select_for_llm(all_transactions, all_=settings.llm_categorize_all)

    rule_enriched: dict[str, Transaction] = {}
    llm_enriched: dict[str, Transaction] = {}
    if to_classify:
        # Шаг 1: правила-справочник (без LLM).
        rule_enriched, llm_remaining = await _resolve_by_rules(
            to_classify,
            family_id=family_id,
            rule_repo=PostgresMerchantRuleRepository(),
            threshold=settings.merchant_match_threshold,
        )
        # Шаг 2: всё, что не узнали по правилам — в LLM.
        if llm_remaining:
            taxonomy = await _load_taxonomy()
            llm_enriched = await _run_llm_batch(
                llm_remaining,
                family_id=family_id,
                system_prompt=build_system_prompt(taxonomy),
            )

    enriched_map = {**rule_enriched, **llm_enriched}

    # Merge enriched into full list — single source of truth for both the
    # state checkpoint AND the clarification builder, so they can't disagree.
    merged = _merge_enriched(all_transactions, enriched_map)

    # Fire-and-forget: ONE Graphiti episode summarising this import (period,
    # total, top categories) — written here, after categories are assigned, so
    # the episode carries real category names. One aggregate episode per import
    # (~4 LLM calls), not per row, keeps bulk imports cheap (see ingest_node).
    _write_import_episode(family_id, merged)

    # Production-скор для дашборда: доля операций, ушедших на ручную проверку
    # среди тех, что мы пытались классифицировать (rule + LLM).
    if to_classify:
        attempted_keys = {tx.import_hash or str(tx.transaction_id) for tx in to_classify}
        by_key = {tx.import_hash or str(tx.transaction_id): tx for tx in merged}
        reviewed = sum(1 for k in attempted_keys if by_key[k].needs_review)
        emit_score(
            "categorization_review_rate",
            reviewed / len(to_classify),
            comment=f"{reviewed}/{len(to_classify)} needs_review",
        )

    # Rebuild clarification questions from post-cascade state
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
    if rule_enriched:
        parts.append(f"Узнал по справочнику без LLM: {len(rule_enriched)}.")
    if llm_enriched:
        classified_count = sum(
            1 for tx in llm_enriched.values() if tx.category != Category.UNCLASSIFIED
        )
        parts.append(f"LLM категоризировал {classified_count}/{len(llm_enriched)} транзакций.")
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


def _write_import_episode(family_id: uuid.UUID, transactions: list[Transaction]) -> None:
    """Schedule the fire-and-forget import-summary episode (no-op if no rows)."""
    if not transactions:
        return
    expenses = [tx for tx in transactions if tx.direction == Direction.EXPENSE]
    if not expenses:
        return
    reference_time = max(tx.occurred_at for tx in expenses)
    task = asyncio.create_task(
        add_episode(
            name=make_import_episode_name(transactions),
            body=import_to_episode_body(transactions),
            source_description="bank_import",
            reference_time=reference_time,
            group_id=str(family_id),
        )
    )
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _select_for_llm(
    transactions: list[Transaction],
    *,
    all_: bool,
) -> list[Transaction]:
    if all_:
        return [tx for tx in transactions if tx.direction != Direction.TRANSFER]
    return [tx for tx in transactions if tx.category == Category.UNCLASSIFIED or tx.needs_review]


_CATEGORIZER_CONCURRENCY = 10  # max parallel LLM calls; keeps OpenRouter happy

# confidence для попаданий по правилу-справочнику: высоко, но < 1.0 (не «истина»,
# а уверенное эвристическое сопоставление) — оставляем место ручному правилу/правке.
_RULE_CONFIDENCE = 0.95


async def _load_taxonomy() -> str:
    """Подтянуть таксономию из справочника; при ошибке — пустая строка (fallback)."""
    try:
        return await PostgresCategoryCatalog().render_taxonomy()
    except Exception:
        logger.exception("categorizer_taxonomy_load_failed")
        return ""


async def _resolve_by_rules(
    transactions: list[Transaction],
    *,
    family_id: uuid.UUID,
    rule_repo: MerchantRuleRepository,
    threshold: float,
) -> tuple[dict[str, Transaction], list[Transaction]]:
    """Каскад, шаг 1: назначить категории по правилам-справочнику (без LLM).

    Возвращает (enriched_map для попаданий, список оставшихся для LLM).
    Промахи и ошибки поиска отправляют транзакцию дальше в LLM (мягкая деградация).
    """
    try:
        hits: dict[str, MerchantRuleHit] = await rule_repo.lookup_many(
            family_id=family_id,
            merchants=[tx.merchant_raw for tx in transactions],
            threshold=threshold,
        )
    except Exception:
        logger.exception("categorizer_rule_lookup_failed")
        return {}, transactions

    enriched: dict[str, Transaction] = {}
    remaining: list[Transaction] = []
    persist_groups: dict[tuple[Category, Direction], list[str]] = {}
    for tx in transactions:
        hit = hits.get(tx.merchant_raw)
        if hit is None:
            remaining.append(tx)
            continue
        direction = direction_for_category(hit.category)
        enriched[tx.import_hash or str(tx.transaction_id)] = tx.model_copy(
            update={
                "category": hit.category,
                "direction": direction,
                "confidence": _RULE_CONFIDENCE,
                "needs_review": False,
            }
        )
        if tx.import_hash:
            persist_groups.setdefault((hit.category, direction), []).append(tx.import_hash)

    if persist_groups:
        repository = PostgresTransactionRepository()
        for (category, direction), hashes in persist_groups.items():
            await repository.classify_by_import_hashes(
                family_id=family_id,
                import_hashes=hashes,
                category=category,
                direction=direction,
                confidence=_RULE_CONFIDENCE,
                needs_review=False,
            )

    return enriched, remaining


async def _categorize_one(
    tx: Transaction,
    model: Runnable[LanguageModelInput, CategoryPrediction],
    sem: asyncio.Semaphore,
    system_prompt: str,
) -> Transaction | None:
    """Categorize a single transaction under a concurrency semaphore.

    Returns ``None`` on LLM failure so the caller skips a pointless re-write of
    the unchanged (UNCLASSIFIED) row.
    """
    async with sem:
        try:
            prediction = await model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Продавец: {tx.merchant_raw}\nСумма: {tx.amount} ₽"),
                ]
            )
            return tx.model_copy(
                update={
                    "category": prediction.category,
                    "direction": direction_for_category(prediction.category),
                    "confidence": prediction.confidence,
                    "needs_review": prediction.confidence < 0.7,
                }
            )
        except Exception:
            logger.exception("categorizer_llm_failed", merchant=tx.merchant_raw)
            return None


async def _run_llm_batch(
    transactions: list[Transaction],
    *,
    family_id: uuid.UUID,
    system_prompt: str,
) -> dict[str, Transaction]:
    """Run LLM on all transactions concurrently (bounded semaphore); return enriched map."""
    model = cast(
        "Runnable[LanguageModelInput, CategoryPrediction]",
        get_chat_model(tier="worker").with_structured_output(CategoryPrediction),
    )
    repository = PostgresTransactionRepository()
    sem = asyncio.Semaphore(_CATEGORIZER_CONCURRENCY)

    results = await asyncio.gather(
        *(_categorize_one(tx, model, sem, system_prompt) for tx in transactions)
    )
    # Drop failures (None): keep originals untouched and skip their DB re-write.
    enriched = [tx for tx in results if tx is not None]

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
