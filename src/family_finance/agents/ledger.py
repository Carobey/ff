"""Ledger agent: answer simple spending questions from persisted transactions.

Flow (P1-18):
  1. parse_ledger_query() — deterministic intent parse (category + period)
  2. aggregate() × 2 — current period + same period last month
  3. _narrative_llm() — LLM turns numbers into a natural sentence
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from family_finance.agents._messages import message_text
from family_finance.agents.ledger_terms import CATEGORY_RULES, MONTH_TOKENS, CategoryRule
from family_finance.agents.state import FinanceState
from family_finance.application.ports import LedgerSummary
from family_finance.domain import Category, Direction
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.mcp import call_finance_tool

logger = structlog.get_logger()


@dataclass(frozen=True)
class LedgerQuery:
    label: str
    categories: tuple[Category, ...]
    directions: tuple[Direction, ...]
    period_label: str
    start: datetime | None = None
    end: datetime | None = None


def is_ledger_question(text: str) -> bool:
    """Detect simple spending questions with deterministic rules."""
    normalized = text.lower()
    return any(word in normalized for word in ("сколько", "потрат", "траты", "расход"))


def parse_ledger_query(
    text: str,
    *,
    current_year: int | None = None,
    now: datetime | None = None,
) -> LedgerQuery | None:
    """Parse a small, explicit subset of Russian finance questions."""
    normalized = text.lower().replace("ё", "е")
    category_match = _match_category(normalized)
    if category_match is None:
        return None

    label, categories, directions = category_match
    start, end, period_label = _parse_period(normalized, current_year=current_year, now=now)
    return LedgerQuery(
        label=label,
        categories=categories,
        directions=directions,
        period_label=period_label,
        start=start,
        end=end,
    )


async def ledger_node(state: FinanceState) -> dict[str, object]:
    """Answer a spending question with a narrative LLM response (P1-18)."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else ""
    query = parse_ledger_query(user_text)
    if query is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Пока понимаю вопросы вроде: "
                        "«сколько на еду в мае?» или «все расходы за апрель»."
                    )
                )
            ],
            "current_intent": "idle",
        }

    family_id = uuid.UUID(state["family_id"])

    # Current period — read through the family-finance MCP server, not the repo
    current = await _aggregate_via_mcp(family_id, query, query.start, query.end)

    # Previous period (same length) for comparison
    prev_summary: LedgerSummary | None = None
    if query.start and query.end:
        delta = query.end - query.start
        prev_start = query.start - delta
        prev_end = query.start
        try:
            prev_summary = await _aggregate_via_mcp(family_id, query, prev_start, prev_end)
        except Exception:
            logger.warning("ledger_prev_period_failed")

    narrative = await _narrative_llm(
        user_question=user_text,
        query=query,
        current=current,
        previous=prev_summary,
    )
    return {
        "messages": [AIMessage(content=narrative)],
        "current_intent": "idle",
    }


async def _aggregate_via_mcp(
    family_id: uuid.UUID,
    query: LedgerQuery,
    start: datetime | None,
    end: datetime | None,
) -> LedgerSummary:
    """Aggregate one period through the ``aggregate_spending`` MCP tool."""
    data = await call_finance_tool(
        "aggregate_spending",
        {
            "family_id": str(family_id),
            "categories": [c.value for c in query.categories],
            "directions": [d.value for d in query.directions],
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
    )
    return LedgerSummary(total=Decimal(str(data["total"])), count=int(data["count"]))


_NARRATIVE_SYSTEM = """\
Ты — финансовый помощник семьи. Отвечай коротко и конкретно (1-3 предложения) по-русски.

Правила:
- Сумму пиши с пробелами-разделителями тысяч: «28 881 ₽», а не «28881₽»
- Если есть данные за предыдущий период — сравни: «на 12% меньше, чем в апреле»
- Если данных нет (count=0) — скажи что транзакций не найдено за этот период
- Не придумывай числа — используй только те, что даны
"""


async def _narrative_llm(
    *,
    user_question: str,
    query: LedgerQuery,
    current: LedgerSummary,
    previous: LedgerSummary | None,
) -> str:
    """One LLM call to turn aggregate numbers into a natural narrative."""
    prev_block = ""
    if previous is not None and previous.count > 0:
        pct = _pct_change(current.total, previous.total) if previous.total != Decimal("0") else None
        prev_block = f"\nПредыдущий период: {previous.total:.2f} ₽, {previous.count} операций."
        if pct is not None:
            prev_block += f" Изменение: {pct:+.1f}%."

    facts = (
        f"Вопрос пользователя: «{user_question}»\n"
        f"Категория запроса: {query.label}\n"
        f"Период: {query.period_label}\n"
        f"Текущий период: {current.total:.2f} ₽, {current.count} операций."
        f"{prev_block}"
    )

    try:
        model = get_chat_model(tier="worker")
        response = await model.ainvoke(
            [SystemMessage(content=_NARRATIVE_SYSTEM), HumanMessage(content=facts)],
        )
        return message_text(response)
    except Exception:
        logger.exception("ledger_narrative_llm_failed")
        # Graceful fallback — still useful without narrative
        return (
            f"{query.label.capitalize()} {query.period_label}: "
            f"{current.total:,.2f} ₽, операций: {current.count}."
        )


def _pct_change(current: Decimal, previous: Decimal) -> float:
    if previous == Decimal("0"):
        return 0.0
    return float((current - previous) / previous * 100)


def _match_category(
    normalized: str,
) -> tuple[str, tuple[Category, ...], tuple[Direction, ...]] | None:
    for rule in CATEGORY_RULES:
        if _matches_rule(normalized, rule):
            return rule.label, rule.categories, rule.directions
    return None


def _matches_rule(normalized: str, rule: CategoryRule) -> bool:
    return any(token in normalized for token in rule.tokens)


def _parse_period(
    normalized: str,
    *,
    current_year: int | None,
    now: datetime | None,
) -> tuple[datetime | None, datetime | None, str]:
    effective_now = now or datetime.now(UTC)
    year = current_year or effective_now.year

    date_range = _parse_day_range(normalized, year=year)
    if date_range is not None:
        return date_range

    if any(token in normalized for token in ("этот месяц", "текущий месяц")):
        start = _month_start(effective_now.year, effective_now.month)
        end = _next_month_start(effective_now.year, effective_now.month)
        return start, end, f"за {effective_now.month:02d}.{effective_now.year}"

    if "прошлый месяц" in normalized:
        previous_year, previous_month = _previous_month(effective_now.year, effective_now.month)
        start = _month_start(previous_year, previous_month)
        end = _next_month_start(previous_year, previous_month)
        return start, end, f"за {previous_month:02d}.{previous_year}"

    month = _find_month(normalized)
    if month is not None:
        start = _month_start(year, month)
        end = _next_month_start(year, month)
        return start, end, f"за {month:02d}.{year}"
    return None, None, "за все время"


def _parse_day_range(
    normalized: str,
    *,
    year: int,
) -> tuple[datetime, datetime, str] | None:
    match = re.search(
        r"\bс\s+(\d{1,2})\s+по\s+(\d{1,2})\s+([а-я]+)(?:\s+(\d{4}))?",
        normalized,
    )
    if match is None:
        return None

    start_day = int(match.group(1))
    end_day = int(match.group(2))
    month_word = match.group(3)
    range_year = int(match.group(4)) if match.group(4) else year
    month = _find_month(month_word)
    if month is None:
        return None

    try:
        start = datetime(range_year, month, start_day, tzinfo=UTC)
        end_inclusive = datetime(range_year, month, end_day, tzinfo=UTC)
    except ValueError:
        return None

    if end_inclusive < start:
        return None

    end = end_inclusive + timedelta(days=1)
    return (
        start,
        end,
        f"с {start_day:02d}.{month:02d}.{range_year} по {end_day:02d}.{month:02d}.{range_year}",
    )


def _find_month(normalized: str) -> int | None:
    for tokens, month in MONTH_TOKENS:
        if any(token in normalized for token in tokens):
            return month
    return None


def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _next_month_start(year: int, month: int) -> datetime:
    return datetime(year + (month // 12), (month % 12) + 1, 1, tzinfo=UTC)


def _previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1
