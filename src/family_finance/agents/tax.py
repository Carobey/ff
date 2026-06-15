"""TaxAgent: оценка возврата НДФЛ по социальным вычетам (ст. 219 НК).

Вычитаемые траты уже разложены категоризатором по префиксу ``tax_ded.*`` и
``charity`` — ноде остаётся отфильтровать их за налоговый год, спросить у юзера
неоднозначные флаги и посчитать возврат. Числа считает Python (домен
``estimate_social_deductions``), LLM не участвует — расчёт детерминирован.

Два пути (как у advisor, PR-03):
- ``tax_node`` — одиночный интент. HITL через ``interrupt()``: спрашивает доход
  (если не виден в транзакциях) и флаги «дорогостоящее лечение / обучение детей»,
  затем считает точно.
- ``build_tax_section`` — воркер веера мульти-интента (ADR 0008). HITL в веере
  невозможен (параллельные ``Send`` без пауз), поэтому даёт детерминированную
  best-effort оценку (всё лечение — обычное, обучение — своё, без детей) с честной
  пометкой и отсылкой к точному расчёту через одиночный запрос.
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal

import structlog
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from family_finance.agents.state import FinanceState, SectionResult
from family_finance.domain import (
    Category,
    DeductionEstimate,
    DeductionInput,
    Direction,
    estimate_social_deductions,
)
from family_finance.infrastructure.mcp import MCPLedgerReader

logger = structlog.get_logger()

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

_INCOME_CATEGORIES = (Category.INCOME_SALARY, Category.INCOME_OTHER)

# ── Routing ───────────────────────────────────────────────────────────────────

_TAX_TOKENS = (
    "вычет",  # вычет/вычеты/вычета/вычетов
    "ндфл",
    "налоговый возврат",
    "вернуть налог",
    "возврат налога",
    "налоговая декларация",
    "3-ндфл",
)


def is_tax_question(text: str) -> bool:
    """Detect tax-deduction questions for the TaxAgent."""
    normalized = text.lower().replace("ё", "е")
    return any(tok in normalized for tok in _TAX_TOKENS)


# ── Сбор вычитаемых трат ──────────────────────────────────────────────────────


class _Deductible:
    """Суммы вычитаемых трат за период (по категориям, ст. 219)."""

    __slots__ = ("charity", "education", "medical", "sport")

    def __init__(self) -> None:
        self.medical = Decimal("0")  # tax_ded.medical (общий лимит / дорогостоящее)
        self.education = Decimal("0")  # tax_ded.education (своё / дети)
        self.sport = Decimal("0")  # tax_ded.sport (общий лимит)
        self.charity = Decimal("0")  # charity (25% дохода)

    @property
    def total(self) -> Decimal:
        return self.medical + self.education + self.sport + self.charity


def _collect(breakdown: list[tuple[Category, Decimal, int]]) -> _Deductible:
    """Разнести breakdown по корзинам вычета. ИИС/недвижимость — не соц., игнор."""
    out = _Deductible()
    for category, amount, _ in breakdown:
        if category == Category.TAX_DED_MEDICAL:
            out.medical += amount
        elif category == Category.TAX_DED_EDUCATION:
            out.education += amount
        elif category == Category.TAX_DED_SPORT:
            out.sport += amount
        elif category == Category.CHARITY:
            out.charity += amount
    return out


def _year_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Налоговый период — календарный год: [1 января .. now]."""
    start = datetime(now.year, 1, 1, tzinfo=_MOSCOW)
    return start, now


async def _load(
    family_id: uuid.UUID,
    *,
    repo: MCPLedgerReader,
    start: datetime,
    end: datetime,
) -> tuple[_Deductible, Decimal]:
    """Вернуть вычитаемые траты и годовой доход за период."""
    breakdown = await repo.category_breakdown(family_id=family_id, start=start, end=end)
    deductible = _collect(breakdown)
    income = (
        await repo.aggregate(
            family_id=family_id,
            categories=_INCOME_CATEGORIES,
            directions=(Direction.INCOME,),
            start=start,
            end=end,
        )
    ).total
    return deductible, income


# ── HITL-уточнения (одиночный интент) ─────────────────────────────────────────


def _questions(deductible: _Deductible, income: Decimal, year: int) -> dict[str, object] | None:
    """Что нужно спросить у юзера перед точным расчётом (None — спрашивать нечего).

    Бот форматирует этот dict в форму; ответ приходит как ``Command(resume=...)``.
    """
    need_income = income <= 0
    ask_medical = deductible.medical > 0
    ask_children = deductible.education > 0
    if not (need_income or ask_medical or ask_children):
        return None
    return {
        "kind": "tax_deduction_input",
        "year": year,
        "need_income": need_income,
        "ask_medical_expensive": ask_medical,
        "medical_total": str(deductible.medical),
        "ask_children_education": ask_children,
        "education_total": str(deductible.education),
    }


def _build_input(
    deductible: _Deductible,
    income: Decimal,
    answers: object,
) -> DeductionInput:
    """Собрать вход калькулятора, разнеся медицину/обучение по ответам юзера.

    ``answers`` — dict из resume (``annual_income``, ``medical_expensive``,
    ``education_children``, ``children_count``). Не-dict / пропуски трактуем
    консервативно: доход из транзакций, флаги = 0 (всё обычное/своё).
    """
    data = answers if isinstance(answers, dict) else {}

    annual_income = income
    if income <= 0:
        annual_income = _as_decimal(data.get("annual_income"))

    expensive = _clamp(_as_decimal(data.get("medical_expensive")), deductible.medical)
    children_edu = _clamp(_as_decimal(data.get("education_children")), deductible.education)
    children_count = _as_int(data.get("children_count"))

    return DeductionInput(
        annual_income=annual_income,
        medical_regular=deductible.medical - expensive,
        medical_expensive=expensive,
        education_self=deductible.education - children_edu,
        education_children=children_edu,
        children_count=children_count,
        sport=deductible.sport,
        charity=deductible.charity,
    )


def _as_decimal(value: object) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        result = Decimal(str(value))
    except (ValueError, ArithmeticError):
        return Decimal("0")
    return result if result > 0 else Decimal("0")


def _as_int(value: object) -> int:
    try:
        return max(int(str(value)), 0)
    except (ValueError, TypeError):
        return 0


def _clamp(value: Decimal, upper: Decimal) -> Decimal:
    return min(value, upper)


# ── Node (одиночный интент) ───────────────────────────────────────────────────


async def tax_node(state: FinanceState) -> dict[str, object]:
    """Точная оценка вычета: читает траты за год, при необходимости спрашивает
    доход и флаги через ``interrupt()``, считает возврат детерминированно.

    Узел переисполняется на resume (как ingest, ADR 0009) — чтение трат/дохода
    выше ``interrupt()`` идемпотентно (без сайд-эффектов), запись в БД отсутствует
    вовсе, поэтому повтор безопасен.
    """
    family_id = uuid.UUID(state["family_id"])
    repo = MCPLedgerReader()
    now = datetime.now(tz=_MOSCOW)
    start, end = _year_bounds(now)

    deductible, income = await _load(family_id, repo=repo, start=start, end=end)
    if deductible.total <= 0:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"За {now.year} год не вижу трат, дающих право на социальный "
                        "вычет (лечение, обучение, спорт, ДМС, благотворительность). "
                        "Загрузи выписку — и я посчитаю возможный возврат НДФЛ."
                    )
                )
            ],
            "current_intent": "idle",
        }

    questions = _questions(deductible, income, now.year)
    answers: object = interrupt(questions) if questions is not None else {}

    estimate = estimate_social_deductions(_build_input(deductible, income, answers))
    return {
        "messages": [AIMessage(content=_format_estimate(estimate, year=now.year))],
        "current_intent": "idle",
    }


# ── Orchestrator section (ADR 0008, без HITL) ─────────────────────────────────


async def build_tax_section(
    family_id: uuid.UUID,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    now: datetime | None = None,
) -> SectionResult:
    """Best-effort оценка вычета для веера мульти-интента (без HITL).

    Веер дёргает воркеры параллельно через ``Send`` — паузы ``interrupt()`` тут
    невозможны, поэтому флаги не спрашиваем: считаем всё лечение обычным, обучение
    своим, без детей. Доход берём из транзакций (нет → ставка 13% по умолчанию).
    Для точного расчёта — одиночный запрос «налоговый вычет» через ``tax_node``.
    """
    now = now or datetime.now(tz=_MOSCOW)
    period_start, period_end = (start, end) if start and end else _year_bounds(now)
    repo = MCPLedgerReader()

    deductible, income = await _load(family_id, repo=repo, start=period_start, end=period_end)
    if deductible.total <= 0:
        body = "Трат с правом на социальный вычет за период не нашёл."
        return {
            "kind": "tax",
            "order": 5,
            "title": "Налоговый вычет",
            "body": f"🧾 <b>Налоговый вычет</b>\n{body}",
        }

    estimate = estimate_social_deductions(
        DeductionInput(
            annual_income=income,
            medical_regular=deductible.medical,
            medical_expensive=Decimal("0"),
            education_self=deductible.education,
            education_children=Decimal("0"),
            children_count=0,
            sport=deductible.sport,
            charity=deductible.charity if income > 0 else Decimal("0"),
        )
    )
    body = _format_estimate(estimate, year=now.year, best_effort=True)
    return {
        "kind": "tax",
        "order": 5,
        "title": "Налоговый вычет",
        "body": f"🧾 <b>Налоговый вычет</b>\n{body}",
    }


# ── Рендер (детерминированный) ────────────────────────────────────────────────


def _format_estimate(est: DeductionEstimate, *, year: int, best_effort: bool = False) -> str:
    """Человекочитаемая оценка возврата. Дисклеймер обязателен (честность)."""
    rate_pct = int(est.rate * 100)
    lines = [
        f"Оценка возврата НДФЛ за {year} год по социальным вычетам:",
        f"• База вычета: {_money(est.total_base)} (ставка {rate_pct}%)",
    ]
    if est.general_base > 0:
        cap = " — упёрлись в лимит 150 000 ₽" if est.general_capped else ""
        lines.append(f"• Лечение/обучение/спорт: {_money(est.general_base)}{cap}")
    if est.expensive_base > 0:
        lines.append(f"• Дорогостоящее лечение (без лимита): {_money(est.expensive_base)}")
    if est.child_base > 0:
        cap = " — упёрлись в лимит 110 000 ₽/ребёнка" if est.child_capped else ""
        lines.append(f"• Обучение детей: {_money(est.child_base)}{cap}")
    if est.charity_base > 0:
        cap = " — упёрлись в лимит 25% дохода" if est.charity_capped else ""
        lines.append(f"• Благотворительность: {_money(est.charity_base)}{cap}")
    lines.append(f"💸 К возврату ≈ <b>{_money(est.refund)}</b>")
    if best_effort:
        lines.append(
            "⚠️ Грубая оценка: считаю всё лечение обычным, обучение — своим. "
            "Для точного расчёта спроси «налоговый вычет» отдельно."
        )
    lines.append(
        "Это оценка, не гарантия: фактический возврат зависит от уплаченного НДФЛ "
        "и подтверждающих документов."
    )
    return "\n".join(lines)


def _money(value: Decimal) -> str:
    """Format Decimal as ``1 234 ₽`` (space thousands sep, рубли без копеек)."""
    int_part = int(value)
    return f"{int_part:,}".replace(",", " ") + " ₽"
