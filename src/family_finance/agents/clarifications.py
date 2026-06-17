"""Build user-facing clarification questions for imported transactions."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, TypedDict, TypeGuard

from family_finance.domain import Category, Direction, Transaction, normalize_merchant


class ClarificationQuestion(TypedDict):
    """Serializable question stored in LangGraph state."""

    id: int
    reason: str
    merchant_raw: str
    payment_dates: list[str]
    count: int
    total: str
    import_hashes: list[str]
    text: str


@dataclass(frozen=True)
class ClarificationAnswer:
    """Parsed user answer to one clarification question."""

    question: ClarificationQuestion
    raw_label: str
    category: Category
    direction: Direction


@dataclass(frozen=True)
class ClarificationGroup:
    """A compact group of similar transactions that needs user clarification."""

    reason: str
    merchant_raw: str
    payment_dates: tuple[date, ...]
    count: int
    total: Decimal
    import_hashes: tuple[str, ...]


def build_import_questions(
    transactions: list[Transaction],
    *,
    limit: int = 5,
) -> list[ClarificationQuestion]:
    """Return structured questions for transfers and low-confidence categories."""
    groups = _group_transactions(transactions)
    questions = [
        _build_question(index=index, group=group)
        for index, group in enumerate(groups[:limit], start=1)
    ]
    remaining = len(groups) - limit
    if remaining > 0:
        questions.append(
            {
                "id": len(questions) + 1,
                "reason": "summary",
                "merchant_raw": "",
                "payment_dates": [],
                "count": 0,
                "total": "0",
                "import_hashes": [],
                "text": f"Ещё {remaining} групп операций требуют уточнения, покажу позже.",
            }
        )
    return questions


def format_import_questions(questions: list[ClarificationQuestion]) -> list[str]:
    """Return display text for clarification questions."""
    return [question["text"] for question in questions]


def parse_clarification_answers(
    text: str,
    questions: list[ClarificationQuestion] | list[Any],
) -> list[ClarificationAnswer]:
    """Parse numbered answers like `1 одежда 2 коммуналка`."""
    by_id = _questions_by_id(questions)
    answers: list[ClarificationAnswer] = []
    for question_id, raw_label in _extract_numbered_answers(text, set(by_id)):
        question = by_id.get(question_id)
        if question is None:
            continue
        classification = _classify_answer(raw_label)
        if classification is None:
            continue
        category, direction = classification
        answers.append(
            ClarificationAnswer(
                question=question,
                raw_label=raw_label,
                category=category,
                direction=direction,
            )
        )
    return answers


# Ответы-«не знаю»: пользователь не может выбрать категорию → уходим в веб-поиск.
# "__lookup__" — sentinel от inline-кнопки «🔎 Не знаю» (см. bot/handlers).
_DONT_KNOW_TOKENS = (
    "__lookup__",
    "не знаю",
    "незнаю",
    "хз",
    "не в курсе",
    "понятия не имею",
    "без понятия",
    "не помню",
    "найди",
    "поищи",
)


def _is_dont_know(raw_label: str) -> bool:
    normalized = raw_label.lower().replace("ё", "е").strip()
    return any(token in normalized for token in _DONT_KNOW_TOKENS)


def parse_unknown_answers(
    text: str,
    questions: list[ClarificationQuestion] | list[Any],
) -> list[ClarificationQuestion]:
    """Вернуть вопросы, на которые пользователь ответил «не знаю» (→ веб-поиск)."""
    by_id = _questions_by_id(questions)
    unknown: list[ClarificationQuestion] = []
    seen: set[int] = set()
    for question_id, raw_label in _extract_numbered_answers(text, set(by_id)):
        if question_id in seen or not _is_dont_know(raw_label):
            continue
        question = by_id.get(question_id)
        if question is not None:
            unknown.append(question)
            seen.add(question_id)
    return unknown


def parse_freetext_answers(
    text: str,
    questions: list[ClarificationQuestion] | list[Any],
) -> list[tuple[ClarificationQuestion, str]]:
    """Вернуть ответы свободным текстом (кнопки не подошли) → LLM-категоризация.

    Это нумерованные ответы, чей текст НЕ известное ключевое слово и НЕ «не знаю»
    (например «1 спортзал»). Их отдаём категоризатору поверх полной таксономии —
    так ручной ввод работает, даже если ни одна кнопка не угадала.
    """
    by_id = _questions_by_id(questions)
    result: list[tuple[ClarificationQuestion, str]] = []
    seen: set[int] = set()
    for question_id, raw_label in _extract_numbered_answers(text, set(by_id)):
        if question_id in seen:
            continue
        question = by_id.get(question_id)
        if question is None:
            continue
        # Уже обрабатывается другими ветками: точная категория или «не знаю».
        if _is_dont_know(raw_label) or _classify_answer(raw_label) is not None:
            continue
        result.append((question, raw_label))
        seen.add(question_id)
    return result


def _is_valid_question(value: object) -> TypeGuard[ClarificationQuestion]:
    """Narrow an opaque state value to a usable ClarificationQuestion."""
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("id"), int)
        and isinstance(value.get("import_hashes"), list)
        and bool(value.get("import_hashes"))
    )


def _questions_by_id(
    questions: list[ClarificationQuestion] | list[Any],
) -> dict[int, ClarificationQuestion]:
    return {q["id"]: q for q in questions if _is_valid_question(q)}


def _group_transactions(transactions: list[Transaction]) -> list[ClarificationGroup]:
    # Ключ группировки — нормализованный мерчант, иначе «Виталий К.» и «ВИТАЛИЙ К.»
    # дают два отдельных вопроса об одном получателе в рамках одного импорта (QA-11).
    # Для отображения берём сырое имя первой операции группы.
    grouped: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
    for tx in transactions:
        reason = _question_reason(tx)
        if reason is None:
            continue
        grouped[(reason, normalize_merchant(tx.merchant_raw))].append(tx)

    result = [
        ClarificationGroup(
            reason=reason,
            merchant_raw=items[0].merchant_raw,
            payment_dates=tuple(sorted({tx.posted_at or tx.occurred_at.date() for tx in items})),
            count=len(items),
            total=sum((tx.amount for tx in items), Decimal("0")),
            import_hashes=tuple(tx.import_hash for tx in items if tx.import_hash is not None),
        )
        for (reason, _merchant_key), items in grouped.items()
    ]
    return sorted(result, key=lambda item: (item.reason, item.merchant_raw))


def _question_reason(tx: Transaction) -> str | None:
    if tx.direction == Direction.TRANSFER:
        return "перевод"
    if tx.category == Category.UNCLASSIFIED or tx.needs_review:
        return "непонятная категория"
    return None


def _build_question(*, index: int, group: ClarificationGroup) -> ClarificationQuestion:
    return {
        "id": index,
        "reason": group.reason,
        "merchant_raw": group.merchant_raw,
        "payment_dates": [item.isoformat() for item in group.payment_dates],
        "count": group.count,
        "total": str(group.total),
        "import_hashes": list(group.import_hashes),
        "text": _format_question(group),
    }


def _format_question(group: ClarificationGroup) -> str:
    amount = f"{group.total:.2f} ₽"
    dates = _format_dates(group.payment_dates)
    if group.count == 1:
        return f"{group.reason.capitalize()}: {dates}, «{group.merchant_raw}», {amount}. Что это?"
    return (
        f"{group.reason.capitalize()}: «{group.merchant_raw}», "
        f"{dates}, {group.count} операций на {amount}. Что это?"
    )


def _format_dates(dates: tuple[date, ...]) -> str:
    if len(dates) == 1:
        return dates[0].strftime("%d.%m.%Y")
    shown = ", ".join(item.strftime("%d.%m.%Y") for item in dates[:3])
    remaining = len(dates) - 3
    if remaining > 0:
        return f"даты: {shown} и ещё {remaining}"
    return f"даты: {shown}"


def _extract_numbered_answers(text: str, valid_ids: set[int]) -> list[tuple[int, str]]:
    # Only treat a number as a question marker when it matches a known question id.
    # Otherwise an answer like "1 потратил 500 на еду" would split on 500 and
    # truncate question 1's label to "потратил".
    matches = [
        m for m in re.finditer(r"(?:^|\s)(\d+)[\).]?\s+", text) if int(m.group(1)) in valid_ids
    ]
    result: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = text[start:end].strip(" ,;")
        if not label:
            continue
        result.append((int(match.group(1)), label))
    return result


def _classify_answer(raw_label: str) -> tuple[Category, Direction] | None:
    # 1. Direct enum value (dot-notation from inline buttons, e.g. "home.utilities")
    try:
        cat = Category(raw_label.strip())
        direction = Direction.TRANSFER if cat == Category.TRANSFER_INTERNAL else Direction.EXPENSE
        if cat in (Category.INCOME_SALARY, Category.INCOME_OTHER):
            direction = Direction.INCOME
        return cat, direction
    except ValueError:
        pass

    # 2. Russian free-text fallback (typed answers).
    # Намеренно отдельная таблица (см. подробный комментарий в ``ledger_terms``):
    # ответ-уточнение → ОДНА (категория, направление). «магазин» здесь =
    # SHOPPING_GENERIC (в ``budgets`` тот же токен ведёт в FOOD_GROCERIES) —
    # конфликт интентов, поэтому таблицы не сливаются.
    normalized = raw_label.lower().replace("ё", "е")
    expense = Direction.EXPENSE
    if any(token in normalized for token in ("одежд", "шопинг")):
        return Category.SHOPPING_CLOTHES, expense
    if any(token in normalized for token in ("коммун", "жкх", "мусор", "вывоз")):
        return Category.HOME_UTILITIES, expense
    if any(token in normalized for token in ("супермаркет", "продукт", "еда", "пикник")):
        return Category.FOOD_GROCERIES, expense
    if any(token in normalized for token in ("ресторан", "кафе", "столовая")):
        return Category.FOOD_RESTAURANT, expense
    if any(token in normalized for token in ("подпис", " ии", "ai", "нейросет")):
        return Category.ENTERTAINMENT_SUBS, expense
    if any(token in normalized for token in ("аптек", "лекарств")):
        return Category.HEALTH_PHARMACY, expense
    _health = ("врач", "клиник", "стрижк", "салон", "красот", "здоровь")
    if any(token in normalized for token in _health):
        return Category.HEALTH_GENERIC, expense
    if any(token in normalized for token in ("перевод", "между своими")):
        return Category.TRANSFER_INTERNAL, Direction.TRANSFER
    if any(token in normalized for token in ("покупк", "магазин", "маркетплейс")):
        return Category.SHOPPING_GENERIC, expense
    return None
