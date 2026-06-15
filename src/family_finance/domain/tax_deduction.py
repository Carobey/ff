"""Социальные налоговые вычеты (ст. 219 НК РФ) — чистый расчёт возврата НДФЛ.

Pure-домен: ни LangGraph, ни БД, ни LLM — только ``Decimal`` и доменные правила
на 2025–2026. «LLM формулирует, Python считает»: суммы и возврат считаются здесь
детерминированно, поэтому eval воспроизводим, а числа не «плывут» от запуска.

Охват — социальные вычеты, выводимые из категоризированных трат:
- общий лимит **150 000 ₽/год** делят: обычное лечение + ДМС, своё обучение, спорт;
- **дорогостоящее лечение** — без лимита (код услуги «2» в справке, флаг от юзера);
- **обучение детей** — отдельный лимит 110 000 ₽ на ребёнка;
- **благотворительность** — до 25% годового дохода.

Возврат = ставка НДФЛ × база (в пределах лимитов). Ставка — маржинальная по
прогрессивной шкале 2025 (доход определяет, по какой ставке считается возврат).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Лимиты (НК РФ, расходы с 2024) ────────────────────────────────────────────
SOCIAL_GENERAL_LIMIT = Decimal("150000")  # общий годовой лимит соцвычета, ст. 219
CHILD_EDUCATION_LIMIT_PER_CHILD = Decimal("110000")  # обучение ребёнка, ст. 219
CHARITY_INCOME_SHARE = Decimal("0.25")  # благотворительность — до 25% дохода

# ── Прогрессивная шкала НДФЛ 2025: (верхняя граница годового дохода, ставка) ───
# Маржинальная ставка: возврат считается по ставке того слоя дохода, который
# уменьшает вычет (для типичной семьи доход ≤ 2.4 млн → 13%).
_NDFL_BRACKETS: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal("2400000"), Decimal("0.13")),
    (Decimal("5000000"), Decimal("0.15")),
    (Decimal("20000000"), Decimal("0.18")),
    (Decimal("50000000"), Decimal("0.20")),
)
_NDFL_TOP_RATE = Decimal("0.22")  # доход свыше 50 млн ₽/год


def ndfl_marginal_rate(annual_income: Decimal) -> Decimal:
    """Маржинальная ставка НДФЛ по годовому доходу (прогрессия 2025)."""
    for ceiling, rate in _NDFL_BRACKETS:
        if annual_income <= ceiling:
            return rate
    return _NDFL_TOP_RATE


class DeductionInput(BaseModel):
    """Уже разнесённые по корзинам суммы (разнесение — в ноде по флагам юзера).

    Чистая функция получает суммы готовыми: медицина разделена на обычную и
    дорогостоящую, обучение — на своё и детское. Так расчёт остаётся простым и
    полностью тестируемым, а вся неоднозначность (что дорогостоящее, что детское)
    решается выше — у пользователя через ``interrupt()``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    annual_income: Decimal = Field(ge=0)
    medical_regular: Decimal = Field(ge=0)  # обычное лечение + ДМС → общий лимит
    medical_expensive: Decimal = Field(ge=0)  # дорогостоящее → без лимита
    education_self: Decimal = Field(ge=0)  # своё обучение → общий лимит
    education_children: Decimal = Field(ge=0)  # обучение детей → отдельный лимит
    children_count: int = Field(ge=0, default=0)
    sport: Decimal = Field(ge=0)  # спорт/фитнес → общий лимит
    charity: Decimal = Field(ge=0)  # благотворительность → 25% дохода

    @field_validator(
        "annual_income",
        "medical_regular",
        "medical_expensive",
        "education_self",
        "education_children",
        "sport",
        "charity",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: object) -> Decimal:
        # Деньги — всегда через str, никогда из float (PRIME-правило точности).
        return Decimal(str(v))


class DeductionEstimate(BaseModel):
    """Результат расчёта: базы по корзинам + итоговый возврат НДФЛ."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annual_income: Decimal
    rate: Decimal
    general_base: Decimal  # учтено по общему лимиту 150k
    general_capped: bool  # упёрлись в лимит (сумма была больше)
    expensive_base: Decimal  # дорогостоящее лечение (без лимита)
    child_base: Decimal  # обучение детей в пределах 110k×N
    child_capped: bool
    charity_base: Decimal  # благотворительность в пределах 25% дохода
    charity_capped: bool
    total_base: Decimal  # суммарная база вычета
    refund: Decimal  # возврат = rate × total_base, копейки


def estimate_social_deductions(inp: DeductionInput) -> DeductionEstimate:
    """Посчитать возврат НДФЛ по социальным вычетам (детерминированно)."""
    # Общий лимит делят обычное лечение + ДМС, своё обучение, спорт.
    general_raw = inp.medical_regular + inp.education_self + inp.sport
    general_base = min(general_raw, SOCIAL_GENERAL_LIMIT)

    # Дорогостоящее лечение — вне лимита.
    expensive_base = inp.medical_expensive

    # Обучение детей — отдельный лимит 110k на каждого ребёнка.
    child_cap = CHILD_EDUCATION_LIMIT_PER_CHILD * inp.children_count
    child_base = min(inp.education_children, child_cap)

    # Благотворительность — до 25% годового дохода.
    charity_cap = (inp.annual_income * CHARITY_INCOME_SHARE).quantize(Decimal("0.01"))
    charity_base = min(inp.charity, charity_cap)

    total_base = general_base + expensive_base + child_base + charity_base
    rate = ndfl_marginal_rate(inp.annual_income)
    refund = (total_base * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return DeductionEstimate(
        annual_income=inp.annual_income,
        rate=rate,
        general_base=general_base,
        general_capped=general_raw > SOCIAL_GENERAL_LIMIT,
        expensive_base=expensive_base,
        child_base=child_base,
        child_capped=inp.education_children > child_cap,
        charity_base=charity_base,
        charity_capped=inp.charity > charity_cap,
        total_base=total_base,
        refund=refund,
    )
