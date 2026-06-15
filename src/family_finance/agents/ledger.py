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
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, cast

import structlog
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from family_finance.agents._messages import message_text
from family_finance.agents.ledger_terms import (
    CATEGORY_LABELS,
    CATEGORY_RULES,
    MONTH_TOKENS,
    CategoryRule,
)
from family_finance.agents.state import FinanceState, SectionResult
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


def parse_period(
    text: str,
    *,
    current_year: int | None = None,
    now: datetime | None = None,
) -> tuple[datetime | None, datetime | None, str]:
    """Parse just the period window from free text (no category needed).

    Used by the multi-intent planner (ADR 0008) to resolve the shared period
    once, so every section (spending/budgets/advice) reports the same window.
    """
    normalized = text.lower().replace("ё", "е")
    return _parse_period(normalized, current_year=current_year, now=now)


# Все расходы: всё, кроме доходов и переводов между своими счетами.
# UNCLASSIFIED включён намеренно — иначе суммы занижены (требование Юрия).
_ALL_EXPENSE_CATEGORIES: tuple[Category, ...] = tuple(
    c
    for c in Category
    if c
    not in (
        Category.INCOME_SALARY,
        Category.INCOME_OTHER,
        Category.TRANSFER_INTERNAL,
    )
)


def all_expenses_query(
    start: datetime | None,
    end: datetime | None,
    period_label: str,
) -> LedgerQuery:
    """Build an «all expenses» query for a given period (used by the веер)."""
    return LedgerQuery(
        label="все расходы",
        categories=_ALL_EXPENSE_CATEGORIES,
        directions=(Direction.EXPENSE,),
        period_label=period_label,
        start=start,
        end=end,
    )


async def build_spending_section(
    family_id: uuid.UUID,
    *,
    start: datetime | None,
    end: datetime | None,
    period_label: str,
) -> SectionResult:
    """Spending breakdown «по категориям» for the period — one orchestrator section.

    Numbers come from MCP and are rendered deterministically in Python, so the
    section body is safe to drop straight into the synthesized answer.
    """
    query = all_expenses_query(start, end, period_label)
    buckets = await _grouped_via_mcp(family_id, query, "category", limit=20)
    return {
        "kind": "spending",
        "order": 1,
        "title": "Расходы",
        "body": _render_grouped(query, "category", buckets),
    }


class QueryShape(BaseModel):
    """Форма ответа на вопрос о тратах — её выбирает LLM по тексту вопроса.

    LLM выбирает ТОЛЬКО структуру (как группировать / список ли это), но не
    считает числа — суммы рендерятся детерминированно из данных MCP.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["aggregate", "list"] = Field(
        default="aggregate",
        description="'list' — показать отдельные операции; 'aggregate' — суммы.",
    )
    group_by: Literal["day", "week", "month", "category", "merchant", "total"] = Field(
        default="total",
        description=(
            "Как группировать суммы (mode=aggregate). day/week/month — по времени "
            "(«по дням», «помесячно»); category — по категориям; merchant — по "
            "продавцам; total — одно общее число (по умолчанию)."
        ),
    )
    then_by: Literal["day", "week", "month", "category", "merchant"] | None = Field(
        default=None,
        description=(
            "Вторая ось для разбивки «X по Y». Например «по дням по категориям» → "
            "group_by=day, then_by=category. Обычно None."
        ),
    )
    order_by: Literal["date_desc", "amount_desc"] = Field(
        default="date_desc",
        description="Для mode=list: amount_desc для «самых крупных», иначе по дате.",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Сколько строк для mode=list.")


_SHAPE_SYSTEM = """\
Ты разбираешь вопрос пользователя о тратах и выбираешь ФОРМУ ответа.
Не считай числа и не придумывай данные — выбери только параметры запроса.

- «по дням / за каждый день» → group_by=day
- «по неделям» → group_by=week
- «по месяцам / помесячно» → group_by=month
- «по категориям / на что» → group_by=category
- «по продавцам / где / у кого» → group_by=merchant
- общий вопрос «сколько потратил» → group_by=total
- «покажи списком / последние операции / топ N» → mode=list
  (для «крупные/большие» используй order_by=amount_desc)

Двойная разбивка «X по Y» → group_by=X, then_by=Y:
- «по дням по категориям» → group_by=day, then_by=category
- «по месяцам по категориям» → group_by=month, then_by=category
- «по категориям по продавцам» → group_by=category, then_by=merchant
Если второй оси нет — then_by оставь пустым (None).
"""


async def _extract_query_shape(user_question: str) -> QueryShape:
    """LLM picks the answer shape. Any failure → safe default (total aggregate)."""
    try:
        model = cast(
            "Runnable[LanguageModelInput, QueryShape]",
            get_chat_model(tier="worker").with_structured_output(QueryShape),
        )
        return await model.ainvoke(
            [SystemMessage(content=_SHAPE_SYSTEM), HumanMessage(content=user_question)]
        )
    except Exception:
        logger.exception("ledger_shape_failed")
        return QueryShape()


# Триггеры разбивки/списка. Если в вопросе НЕТ ни одного — это голое «сколько за
# <период>», и мы детерминированно форсим group_by=total (см. _wants_breakdown).
_BREAKDOWN_KEYWORDS: tuple[str, ...] = (
    "по дн",
    "каждый день",
    "по недел",
    "по месяц",
    "помесяч",
    "по категор",
    "на что",
    "по продавц",
    "по магазин",
    "у кого",
    "покажи",
    "списк",
    "последни",
    "топ",
    "крупн",
    "больш",
    "операци",
)


def _wants_breakdown(text: str) -> bool:
    """Есть ли в вопросе явный триггер разбивки или списка операций."""
    normalized = text.lower()
    return any(kw in normalized for kw in _BREAKDOWN_KEYWORDS)


async def ledger_node(state: FinanceState) -> dict[str, object]:
    """Answer a spending question — flexible aggregation or a transaction list.

    Quality split: an LLM picks the *shape* of the answer (group_by / list /
    ordering) from the free-text question, while period and category filter come
    from the deterministic parser and every number is rendered in Python. The
    LLM never emits figures, so the answer can't drift from the data.
    """
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
                        "Пока понимаю вопросы про траты: «сколько на еду в мае?», "
                        "«все расходы за апрель по дням», «топ-5 трат за месяц»."
                    )
                )
            ],
            "current_intent": "idle",
        }

    family_id = uuid.UUID(state["family_id"])
    shape = await _extract_query_shape(user_text)
    if not _wants_breakdown(user_text):
        # Голый «сколько за <период>» — одно число, без дрейфа shape: LLM иногда
        # отдаёт group_by=category (таблицу) вместо ответа. Форсим total (QA-02).
        shape = QueryShape()

    if shape.mode == "list":
        entries = await _list_via_mcp(family_id, query, shape.order_by, shape.limit)
        return {
            "messages": [AIMessage(content=_render_list(query, entries))],
            "current_intent": "idle",
        }

    # «total + вторая ось» — деградируем к одномерной разбивке по второй оси.
    group_by = shape.group_by
    then_by = shape.then_by
    if group_by == "total" and then_by is not None:
        group_by, then_by = then_by, None

    if group_by == "total":
        current = await _total_via_mcp(family_id, query, query.start, query.end)
        if current.count == 0:
            # Нет данных за период — детерминированный ответ без сравнения и LLM,
            # иначе модель выдаёт бессмыслицу вроде «на 100% меньше».
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"{query.label.capitalize()} {query.period_label}: "
                            "транзакций не найдено."
                        )
                    )
                ],
                "current_intent": "idle",
            }
        prev_summary: LedgerSummary | None = None
        if query.start and query.end:
            delta = query.end - query.start
            try:
                prev_summary = await _total_via_mcp(
                    family_id, query, query.start - delta, query.start
                )
            except Exception:
                logger.warning("ledger_prev_period_failed")
        narrative = await _narrative_llm(
            user_question=user_text, query=query, current=current, previous=prev_summary
        )
        return {"messages": [AIMessage(content=narrative)], "current_intent": "idle"}

    if then_by is not None:
        buckets_2d = await _grouped_2d_via_mcp(family_id, query, group_by, then_by, shape.limit)
        return {
            "messages": [
                AIMessage(content=_render_grouped_2d(query, group_by, then_by, buckets_2d))
            ],
            "current_intent": "idle",
        }

    buckets = await _grouped_via_mcp(family_id, query, group_by, shape.limit)
    return {
        "messages": [AIMessage(content=_render_grouped(query, group_by, buckets))],
        "current_intent": "idle",
    }


async def _total_via_mcp(
    family_id: uuid.UUID,
    query: LedgerQuery,
    start: datetime | None,
    end: datetime | None,
) -> LedgerSummary:
    """One grand total via the ``query_aggregates`` MCP tool (group_by=total)."""
    rows = await call_finance_tool(
        "query_aggregates",
        {
            "family_id": str(family_id),
            "group_by": "total",
            "categories": [c.value for c in query.categories],
            "directions": [d.value for d in query.directions],
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
    )
    if not rows:
        return LedgerSummary(total=Decimal("0"), count=0)
    row = rows[0]
    return LedgerSummary(total=Decimal(str(row["total"])), count=int(row["count"]))


async def _grouped_via_mcp(
    family_id: uuid.UUID,
    query: LedgerQuery,
    group_by: str,
    limit: int,
) -> list[tuple[str, Decimal, int]]:
    """Grouped sums via ``query_aggregates`` → (bucket, total, count) rows."""
    rows = await call_finance_tool(
        "query_aggregates",
        {
            "family_id": str(family_id),
            "group_by": group_by,
            "categories": [c.value for c in query.categories],
            "directions": [d.value for d in query.directions],
            "start": query.start.isoformat() if query.start else None,
            "end": query.end.isoformat() if query.end else None,
            "limit": limit,
        },
    )
    return [(str(r["bucket"]), Decimal(str(r["total"])), int(r["count"])) for r in rows]


async def _grouped_2d_via_mcp(
    family_id: uuid.UUID,
    query: LedgerQuery,
    group_by: str,
    then_by: str,
    limit: int,
) -> list[tuple[str, str, Decimal]]:
    """Two-dimensional sums via ``query_aggregates`` → (bucket, subbucket, total) rows."""
    rows = await call_finance_tool(
        "query_aggregates",
        {
            "family_id": str(family_id),
            "group_by": group_by,
            "then_by": then_by,
            "categories": [c.value for c in query.categories],
            "directions": [d.value for d in query.directions],
            "start": query.start.isoformat() if query.start else None,
            "end": query.end.isoformat() if query.end else None,
            "limit": limit,
        },
    )
    return [
        (str(r["bucket"]), str(r.get("subbucket") or ""), Decimal(str(r["total"]))) for r in rows
    ]


async def _list_via_mcp(
    family_id: uuid.UUID,
    query: LedgerQuery,
    order_by: str,
    limit: int,
) -> list[dict[str, object]]:
    """Individual transactions via the ``list_transactions`` MCP tool."""
    rows = await call_finance_tool(
        "list_transactions",
        {
            "family_id": str(family_id),
            "categories": [c.value for c in query.categories],
            "directions": [d.value for d in query.directions],
            "start": query.start.isoformat() if query.start else None,
            "end": query.end.isoformat() if query.end else None,
            "order_by": order_by,
            "limit": limit,
        },
    )
    return list(rows)


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
        prev_block = (
            f"\nПредыдущий период: {_fmt_money(_round_rub(previous.total))}, "
            f"{previous.count} операций."
        )
        if pct is not None:
            prev_block += f" Изменение: {pct:+.1f}%."

    facts = (
        f"Вопрос пользователя: «{user_question}»\n"
        f"Категория запроса: {query.label}\n"
        f"Период: {query.period_label}\n"
        f"Текущий период: {_fmt_money(_round_rub(current.total))}, {current.count} операций."
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
            f"{_fmt_money(_round_rub(current.total))}, операций: {current.count}."
        )


# ── Deterministic rendering of grouped / list answers ────────────────────────
# Numbers are formatted in Python (never by the LLM), so the column always adds
# up to the printed total — this is what prevents the "не сходится" failure.

_RU_MONTHS_GEN: tuple[str, ...] = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_RU_MONTHS_NOM: tuple[str, ...] = (
    "",
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
)
_GROUP_TITLE: dict[str, str] = {
    "day": "по дням",
    "week": "по неделям",
    "month": "по месяцам",
    "category": "по категориям",
    "merchant": "по продавцам",
}


def _round_rub(amount: Decimal) -> int:
    """Round to whole rubles (kopecks are noise in a breakdown)."""
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _fmt_money(rubles: int) -> str:
    """Целые рубли с пробелом-разделителем тысяч: «1 000 ₽»."""
    return f"{rubles:,}".replace(",", " ") + " ₽"


def _bucket_label(group_by: str, bucket: str) -> str:
    """Human label for one aggregation bucket key."""
    if group_by == "day":
        d = date.fromisoformat(bucket)
        return f"{d.day} {_RU_MONTHS_GEN[d.month]}"
    if group_by == "week":
        d = date.fromisoformat(bucket)
        return f"неделя с {d.day:02d}.{d.month:02d}"
    if group_by == "month":
        year, month = bucket.split("-")
        return f"{_RU_MONTHS_NOM[int(month)]} {year}"
    if group_by == "category":
        try:
            return CATEGORY_LABELS.get(Category(bucket), bucket)
        except ValueError:
            return bucket
    return bucket  # merchant — raw name


def _render_grouped(
    query: LedgerQuery,
    group_by: str,
    buckets: list[tuple[str, Decimal, int]],
) -> str:
    """Format grouped sums; printed total = sum of printed rows (always consistent)."""
    if not buckets:
        return f"{query.label.capitalize()} {query.period_label}: транзакций не найдено."
    title = _GROUP_TITLE.get(group_by, "")
    lines = [f"{query.label.capitalize()} {query.period_label} — {title}:"]
    total = 0
    for bucket, amount, _count in buckets:
        rubles = _round_rub(amount)
        total += rubles
        lines.append(f"• {_bucket_label(group_by, bucket)}: {_fmt_money(rubles)}")
    lines.append(f"Итого: {_fmt_money(total)}")
    return "\n".join(lines)


def _render_grouped_2d(
    query: LedgerQuery,
    group_by: str,
    then_by: str,
    rows: list[tuple[str, str, Decimal]],
) -> str:
    """Format a 2-D breakdown («X по Y»); every subtotal/total is summed in Python."""
    if not rows:
        return f"{query.label.capitalize()} {query.period_label}: транзакций не найдено."

    # Сохраняем порядок первичных бакетов (строки уже отсортированы в SQL).
    grouped: dict[str, list[tuple[str, int]]] = {}
    for bucket, subbucket, amount in rows:
        grouped.setdefault(bucket, []).append((subbucket, _round_rub(amount)))

    title = _GROUP_TITLE.get(group_by, "")
    sub_title = _GROUP_TITLE.get(then_by, "")
    lines = [f"{query.label.capitalize()} {query.period_label} — {title} {sub_title}:"]
    grand_total = 0
    for bucket, subrows in grouped.items():
        bucket_total = sum(rubles for _sub, rubles in subrows)
        grand_total += bucket_total
        lines.append(f"{_bucket_label(group_by, bucket)} — {_fmt_money(bucket_total)}:")
        for subbucket, rubles in subrows:
            lines.append(f"  • {_bucket_label(then_by, subbucket)}: {_fmt_money(rubles)}")
    lines.append(f"Итого: {_fmt_money(grand_total)}")
    return "\n".join(lines)


def _render_list(query: LedgerQuery, entries: list[dict[str, object]]) -> str:
    """Format individual transactions; printed total = sum of printed rows."""
    if not entries:
        return f"{query.label.capitalize()} {query.period_label}: операций не найдено."
    lines = [f"{query.label.capitalize()} {query.period_label} — операции:"]
    total = 0
    for entry in entries:
        occurred = datetime.fromisoformat(str(entry["occurred_at"]))
        rubles = _round_rub(Decimal(str(entry["amount"])))
        total += rubles
        merchant = str(entry["merchant"]) or "—"
        lines.append(f"• {occurred.day:02d}.{occurred.month:02d} {merchant}: {_fmt_money(rubles)}")
    lines.append(f"Итого показано: {_fmt_money(total)}")
    return "\n".join(lines)


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
