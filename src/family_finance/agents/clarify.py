"""Clarification agent: apply user answers to pending import questions."""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from family_finance.agents.clarifications import parse_clarification_answers
from family_finance.agents.state import FinanceState
from family_finance.infrastructure.persistence import PostgresTransactionRepository


def has_clarification_answers(state: FinanceState, user_text: str) -> bool:
    """Return True when text answers at least one open clarification question."""
    open_questions = state.get("open_questions", [])
    if not open_questions:
        return False
    return bool(parse_clarification_answers(user_text, open_questions))


async def clarify_node(state: FinanceState) -> dict[str, object]:
    """Parse numbered answers and update matching transactions."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else ""
    open_questions = state.get("open_questions", [])
    answers = parse_clarification_answers(user_text, open_questions)
    if not answers:
        return {
            "messages": [
                AIMessage(content="Не понял уточнение. Ответь в формате: `1 одежда 2 коммуналка`.")
            ],
            "current_intent": "idle",
        }

    family_id = uuid.UUID(state["family_id"])
    repository = PostgresTransactionRepository()
    updated = 0
    applied_question_ids: set[int] = set()
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

    remaining_questions = [
        question for question in open_questions if question["id"] not in applied_question_ids
    ]
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Принял уточнения: {len(applied_question_ids)}. Обновил транзакций: {updated}."
                )
            )
        ],
        "open_questions": remaining_questions,
        "current_intent": "idle",
    }
