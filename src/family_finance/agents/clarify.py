"""Clarification agent: apply user answers to pending import questions."""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from family_finance.agents.clarifications import (
    ClarificationQuestion,
    parse_clarification_answers,
    parse_freetext_answers,
    parse_unknown_answers,
)
from family_finance.agents.state import FinanceState
from family_finance.agents.web_lookup import categorize_freetext, lookup_merchant
from family_finance.domain import Category, direction_for_category
from family_finance.infrastructure.persistence import (
    PostgresMerchantRuleRepository,
    PostgresTransactionRepository,
)


def has_clarification_answers(state: FinanceState, user_text: str) -> bool:
    """Return True when text answers at least one open clarification question.

    Covers both a category pick AND a «не знаю» answer (the latter triggers the
    web-lookup branch in :func:`clarify_node`).
    """
    open_questions = state.get("open_questions", [])
    if not open_questions:
        return False
    return bool(
        parse_clarification_answers(user_text, open_questions)
        or parse_unknown_answers(user_text, open_questions)
        or parse_freetext_answers(user_text, open_questions)
    )


async def clarify_node(state: FinanceState) -> dict[str, object]:
    """Parse numbered answers and update matching transactions."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else ""
    open_questions = state.get("open_questions", [])
    answers = parse_clarification_answers(user_text, open_questions)
    unknowns = parse_unknown_answers(user_text, open_questions)
    freetexts = parse_freetext_answers(user_text, open_questions)
    if not answers and not unknowns and not freetexts:
        return {
            "messages": [
                AIMessage(content="Не понял уточнение. Ответь в формате: `1 одежда 2 коммуналка`.")
            ],
            "current_intent": "idle",
        }

    family_id = uuid.UUID(state["family_id"])
    repository = PostgresTransactionRepository()
    rule_repo = PostgresMerchantRuleRepository()
    updated = 0
    applied_question_ids: set[int] = set()
    extra_messages: list[str] = []
    for answer in answers:
        updated += await repository.classify_by_import_hashes(
            family_id=family_id,
            import_hashes=answer.question["import_hashes"],
            category=answer.category,
            direction=answer.direction,
            subcategory_freetext=answer.raw_label,
            needs_review=False,
        )
        applied_question_ids.add(answer.question["id"])
        # Learning loop: запоминаем ответ как правило семьи, чтобы следующий импорт
        # того же продавца категоризировался без LLM и без повторного вопроса.
        merchant_raw = answer.question["merchant_raw"]
        if merchant_raw:
            await rule_repo.upsert(
                family_id=family_id,
                merchant_raw=merchant_raw,
                category=answer.category,
                source="user",
            )

    # «Не знаю» → веб-поиск продавца, категоризация из справочника, запись правила
    # как source='llm' (машинная догадка, не подтверждённая пользователем).
    for question in unknowns:
        if question["id"] in applied_question_ids:
            continue
        updated += await _resolve_unknown_via_web(
            question,
            family_id=family_id,
            repository=repository,
            rule_repo=rule_repo,
            messages=extra_messages,
        )
        applied_question_ids.add(question["id"])

    # Ручной ввод свободным текстом («1 спортзал») → LLM-категоризация поверх
    # полной таксономии. Это путь «кнопки не угадали, пишу своими словами».
    for question, label in freetexts:
        if question["id"] in applied_question_ids:
            continue
        updated += await _resolve_freetext(
            question,
            label,
            family_id=family_id,
            repository=repository,
            rule_repo=rule_repo,
            messages=extra_messages,
        )
        applied_question_ids.add(question["id"])

    remaining_questions = [
        question for question in open_questions if question["id"] not in applied_question_ids
    ]
    summary = f"Принял уточнения: {len(applied_question_ids)}. Обновил транзакций: {updated}."
    content = "\n".join([summary, *extra_messages])
    return {
        "messages": [AIMessage(content=content)],
        "open_questions": remaining_questions,
        "current_intent": "idle",
    }


async def _resolve_unknown_via_web(
    question: ClarificationQuestion,
    *,
    family_id: uuid.UUID,
    repository: PostgresTransactionRepository,
    rule_repo: PostgresMerchantRuleRepository,
    messages: list[str],
) -> int:
    """Найти продавца в интернете, проставить категорию, записать правило (source='llm').

    Возвращает число обновлённых транзакций. Добавляет строку для пользователя в
    ``messages``. При неудаче — оставляет без категории и сообщает об этом.
    """
    merchant_raw = question["merchant_raw"]
    verdict = await lookup_merchant(merchant_raw)
    if verdict is None or verdict.category == Category.UNCLASSIFIED:
        messages.append(f"«{merchant_raw}» — не нашёл в интернете, оставил без категории.")
        return 0

    direction = direction_for_category(verdict.category)
    updated = await repository.classify_by_import_hashes(
        family_id=family_id,
        import_hashes=question["import_hashes"],
        category=verdict.category,
        direction=direction,
        needs_review=False,
    )
    if merchant_raw:
        await rule_repo.upsert(
            family_id=family_id,
            merchant_raw=merchant_raw,
            category=verdict.category,
            source="llm",
        )
    note = f" ({verdict.description})" if verdict.description else ""
    messages.append(f"🔎 «{merchant_raw}»{note} → {verdict.category.value}. Нашёл в интернете.")
    return updated


async def _resolve_freetext(
    question: ClarificationQuestion,
    label: str,
    *,
    family_id: uuid.UUID,
    repository: PostgresTransactionRepository,
    rule_repo: PostgresMerchantRuleRepository,
    messages: list[str],
) -> int:
    """Категоризировать по тексту пользователя, проставить категорию, записать правило.

    Юзер написал своими словами, что это («спортзал»), потому что ни одна кнопка не
    подошла. Финальную Category выводит LLM (``categorize_freetext``), поэтому правило
    пишем как ``source='llm'`` — это машинная интерпретация подсказки, не прямой выбор.
    Возвращает число обновлённых транзакций; при неудаче — оставляет без категории.
    """
    merchant_raw = question["merchant_raw"]
    category = await categorize_freetext(merchant_raw, label)
    if category is None or category == Category.UNCLASSIFIED:
        messages.append(f"«{merchant_raw}» — не понял категорию «{label}», оставил без категории.")
        return 0

    direction = direction_for_category(category)
    updated = await repository.classify_by_import_hashes(
        family_id=family_id,
        import_hashes=question["import_hashes"],
        category=category,
        direction=direction,
        subcategory_freetext=label,
        needs_review=False,
    )
    if merchant_raw:
        await rule_repo.upsert(
            family_id=family_id,
            merchant_raw=merchant_raw,
            category=category,
            source="llm",
        )
    messages.append(f"✍️ «{merchant_raw}» → {category.value} (по твоему описанию: «{label}»).")
    return updated
