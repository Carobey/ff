"""AdvisorAgent: финансовый наставник — советы по экономии и накоплениям.

Две методики, авто-выбор по контексту вопроса:
  * 50/30/20 — диагностика трат: нужды ≤50%, желания ≤30%, накопления ≥20%
    от дохода. Используется для вопросов «куда уходят деньги / на чём
    сэкономить».
  * pay-yourself-first + подушка — для вопросов про накопления и цель
    (команда ``/goal``).

Все советы строятся ТОЛЬКО на посчитанных SQL-агрегатах. Нет данных —
говорим честно. LLM лишь оборачивает цифры в человеческий текст.
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal
from typing import Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from family_finance.agents._messages import message_text
from family_finance.agents.budgets import current_moscow_month
from family_finance.agents.state import FinanceState
from family_finance.domain import Category, Direction, GoalProgress, SavingsGoal
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.mcp import MCPLedgerReader

logger = structlog.get_logger()

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

# 50/30/20 нормативы (доля от дохода).
_NEEDS_NORM = 50
_WANTS_NORM = 30
_SAVINGS_NORM = 20

_INCOME_CATEGORIES = (Category.INCOME_SALARY, Category.INCOME_OTHER)


# ── Категории → корзины 50/30/20 ──────────────────────────────────────────────

_NEEDS: frozenset[Category] = frozenset(
    {
        Category.FOOD_GROCERIES,
        Category.TRANSPORT_FUEL,
        Category.TRANSPORT_PUBLIC,
        Category.TRANSPORT_CARPARTS,
        Category.KIDS_CLOTHES,
        Category.KIDS_SCHOOL,
        Category.HOME_UTILITIES,
        Category.HOME_HOUSEHOLD,
        Category.HOME_REPAIR,
        Category.HEALTH_PHARMACY,
        Category.HEALTH_GENERIC,
        Category.TAX_DED_MEDICAL,
        Category.TAX_DED_EDUCATION,
    }
)

_WANTS: frozenset[Category] = frozenset(
    {
        Category.FOOD_RESTAURANT,
        Category.FOOD_DELIVERY,
        Category.TRANSPORT_TAXI,
        Category.KIDS_TOYS,
        Category.KIDS_ACTIVITIES,
        Category.SHOPPING_CLOTHES,
        Category.SHOPPING_GENERIC,
        Category.HOME_FURNITURE,
        Category.ENTERTAINMENT_SUBS,
        Category.ENTERTAINMENT_EVENTS,
        Category.ENTERTAINMENT_HOBBIES,
        Category.PETS,
        Category.TAX_DED_SPORT,
    }
)


def bucket_of(category: Category) -> Literal["needs", "wants", "other"]:
    """Map a category to its 50/30/20 bucket (``other`` = unclassified/income)."""
    if category in _NEEDS:
        return "needs"
    if category in _WANTS:
        return "wants"
    return "other"


# ── Spending health (50/30/20) ────────────────────────────────────────────────


class SpendingHealth(BaseModel):
    """This month's income split into 50/30/20 buckets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    income: Decimal
    needs: Decimal
    wants: Decimal
    total_expenses: Decimal

    @property
    def has_income(self) -> bool:
        return self.income > 0

    @property
    def savings(self) -> Decimal:
        """Net left over after all expenses (can be negative)."""
        return self.income - self.total_expenses

    def _pct(self, value: Decimal) -> int | None:
        if self.income <= 0:
            return None
        return round(float(value / self.income * 100))

    @property
    def needs_pct(self) -> int | None:
        return self._pct(self.needs)

    @property
    def wants_pct(self) -> int | None:
        return self._pct(self.wants)

    @property
    def savings_pct(self) -> int | None:
        return self._pct(self.savings)


# ── Routing ───────────────────────────────────────────────────────────────────

_ADVICE_TOKENS = (
    "совет",
    "посоветуй",
    "сэконом",
    "сэкономить",
    "экономи",
    "копить",
    "накопи",
    "накоплен",
    "отложить",
    "отклад",
    "наставник",
    "куда уход",
    "финансовый план",
    "план накоплен",
    "50/30/20",
    "503020",
    "оптимизир",
    "урезать",
    "подушк",
)


def is_advice_question(text: str) -> bool:
    """Detect requests for coaching/advice (vs raw ledger queries)."""
    normalized = text.lower().replace("ё", "е")
    return any(tok in normalized for tok in _ADVICE_TOKENS)


# ── Node ──────────────────────────────────────────────────────────────────────

_ADVISOR_SYSTEM = """\
Ты — финансовый наставник семьи. Помогаешь экономить и копить.

Методики (выбери подходящую под вопрос пользователя):
- 50/30/20: нужды ≤50%, желания ≤30%, накопления ≥20% от дохода. Для вопросов
  «куда уходят деньги», «на чём сэкономить».
- Pay-yourself-first: откладывай в начале месяца, до трат; цель-подушка =
  3-6 месяцев расходов. Для вопросов про накопления и цель.

ПРАВИЛА:
- Используй ТОЛЬКО переданные цифры. Не выдумывай суммы и проценты.
- Нет данных (нет дохода/трат) — честно скажи и предложи загрузить выписку.
- 3-5 предложений, по-русски, суммы с пробелами между тысячами («12 300 ₽»).
- Дай 1-2 конкретных действия: что сократить и/или сколько откладывать.
- Ты НЕ лицензированный финансовый советник — это бытовые рекомендации по
  личному бюджету, без инвестиционных советов.
"""


async def advisor_node(state: FinanceState) -> dict[str, object]:
    """Give grounded saving/cutting advice for the family."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else ""
    family_id = uuid.UUID(state["family_id"])
    repo = MCPLedgerReader()
    now = datetime.now(tz=_MOSCOW)

    health = await analyze_spending(family_id, repo=repo, now=now)
    cut_candidates = await _top_wants(family_id, repo=repo, now=now)
    goal = await repo.get_savings_goal(family_id=family_id)
    progress: GoalProgress | None = None
    if goal is not None:
        progress = await _goal_progress(goal, repo=repo, now=now)

    if not health.has_income and health.total_expenses == 0 and goal is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Пока не вижу ни доходов, ни расходов за этот месяц. "
                        "Загрузи выписку банка — и я подскажу, где сэкономить и "
                        "сколько откладывать."
                    )
                )
            ],
            "current_intent": "idle",
        }

    facts = _build_facts(user_text, health, cut_candidates, progress, now)
    try:
        model = get_chat_model(tier="worker")
        response = await model.ainvoke(
            [SystemMessage(content=_ADVISOR_SYSTEM), HumanMessage(content=facts)],
        )
        reply = message_text(response).strip()
    except Exception:
        logger.exception("advisor_llm_failed")
        reply = _fallback(health, progress, now)

    return {
        "messages": [AIMessage(content=reply)],
        "current_intent": "idle",
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────


async def analyze_spending(
    family_id: uuid.UUID,
    *,
    repo: MCPLedgerReader,
    now: datetime,
) -> SpendingHealth:
    """Compute the 50/30/20 split for the current Moscow month."""
    start, end = current_moscow_month(now)
    breakdown = await repo.category_breakdown(family_id=family_id, start=start, end=end)
    income = (
        await repo.aggregate(
            family_id=family_id,
            categories=_INCOME_CATEGORIES,
            directions=(Direction.INCOME,),
            start=start,
            end=end,
        )
    ).total

    needs = Decimal("0")
    wants = Decimal("0")
    total = Decimal("0")
    for category, amount, _ in breakdown:
        total += amount
        bucket = bucket_of(category)
        if bucket == "needs":
            needs += amount
        elif bucket == "wants":
            wants += amount
    return SpendingHealth(income=income, needs=needs, wants=wants, total_expenses=total)


async def _top_wants(
    family_id: uuid.UUID,
    *,
    repo: MCPLedgerReader,
    now: datetime,
    limit: int = 3,
) -> list[tuple[Category, Decimal]]:
    """Largest discretionary ('wants') categories this month — cut candidates."""
    start, end = current_moscow_month(now)
    breakdown = await repo.category_breakdown(family_id=family_id, start=start, end=end)
    wants = [(cat, amount) for cat, amount, _ in breakdown if bucket_of(cat) == "wants"]
    return wants[:limit]


async def _goal_progress(
    goal: SavingsGoal,
    *,
    repo: MCPLedgerReader,
    now: datetime,
) -> GoalProgress:
    """Net savings since the goal was set."""
    saved = await repo.net_cashflow(
        family_id=goal.family_id,
        start=goal.created_at,
        end=now,
    )
    return GoalProgress(goal=goal, saved_so_far=saved)


# ── Fact / text formatting ────────────────────────────────────────────────────


def _build_facts(
    user_text: str,
    health: SpendingHealth,
    cut_candidates: list[tuple[Category, Decimal]],
    progress: GoalProgress | None,
    now: datetime,
) -> str:
    lines = [f"Вопрос пользователя: «{user_text}»", "", "Траты за текущий месяц (50/30/20):"]
    if health.has_income:
        lines.append(f"Доход: {_money(health.income)}")
        lines.append(f"Нужды: {_money(health.needs)} ({health.needs_pct}% — норма ≤{_NEEDS_NORM}%)")
        lines.append(
            f"Желания: {_money(health.wants)} ({health.wants_pct}% — норма ≤{_WANTS_NORM}%)"
        )
        lines.append(
            f"Накопления (доход − расходы): {_money(health.savings)} "
            f"({health.savings_pct}% — норма ≥{_SAVINGS_NORM}%)"
        )
    else:
        lines.append(f"Нужды: {_money(health.needs)}")
        lines.append(f"Желания: {_money(health.wants)}")
        lines.append("Доход в этом месяце не зафиксирован — процент накоплений посчитать нельзя.")

    if cut_candidates:
        lines.append("")
        lines.append("Крупнейшие «желания» (кандидаты на сокращение):")
        lines.extend(f"- {cat.value}: {_money(amount)}" for cat, amount in cut_candidates)

    lines.append("")
    if progress is not None:
        lines.append(_goal_facts(progress, now))
    else:
        lines.append("Цель накопления не задана.")
    return "\n".join(lines)


def _goal_facts(progress: GoalProgress, now: datetime) -> str:
    goal = progress.goal
    head = f"Цель накопления: {_money(goal.target_amount)}"
    if goal.target_date is not None:
        head += f" к {goal.target_date.strftime('%d.%m.%Y')}"
    lines = [
        head,
        f"Накоплено с момента постановки: {_money(progress.saved_so_far)} ({progress.pct}%)",
    ]
    monthly = progress.monthly_needed(now)
    if monthly is not None and not progress.reached:
        lines.append(f"Чтобы успеть, нужно откладывать {_money(monthly)}/мес.")
    on_track = progress.on_track(now)
    if on_track is False:
        lines.append("Сейчас отстаёшь от графика.")
    return "\n".join(lines)


def _fallback(health: SpendingHealth, progress: GoalProgress | None, now: datetime) -> str:
    """Deterministic advice when the LLM is unavailable."""
    parts: list[str] = []
    if health.has_income and health.savings_pct is not None:
        if health.savings_pct >= _SAVINGS_NORM:
            parts.append(
                f"Откладываешь {health.savings_pct}% дохода — это в норме (≥20%). Так держать."
            )
        else:
            parts.append(
                f"Накопления — {health.savings_pct}% дохода (норма ≥20%). "
                f"Желания: {health.wants_pct}% (норма ≤30%) — есть где урезать."
            )
    else:
        parts.append(
            f"За месяц нужды — {_money(health.needs)}, желания — {_money(health.wants)}. "
            "Доходов не вижу — загрузи выписку, чтобы посчитать процент накоплений."
        )
    if progress is not None:
        monthly = progress.monthly_needed(now)
        tail = f"Цель: накоплено {progress.pct}%."
        if monthly is not None and not progress.reached:
            tail += f" Откладывай {_money(monthly)}/мес."
        parts.append(tail)
    return " ".join(parts)


# ── Digest block (no LLM) ─────────────────────────────────────────────────────


async def build_advice_block(family_id: uuid.UUID, *, now: datetime | None = None) -> str | None:
    """Compact deterministic advisor block for the weekly digest.

    Returns ``None`` when there's nothing to say (no income, no expenses,
    no goal) so the digest doesn't carry an empty section.
    """
    now = now or datetime.now(tz=_MOSCOW)
    repo = MCPLedgerReader()
    health = await analyze_spending(family_id, repo=repo, now=now)
    goal = await repo.get_savings_goal(family_id=family_id)
    progress = await _goal_progress(goal, repo=repo, now=now) if goal is not None else None

    if not health.has_income and health.total_expenses == 0 and goal is None:
        return None

    lines = ["💡 <b>Наставник</b>"]
    if health.has_income and health.savings_pct is not None:
        verdict = "в норме" if health.savings_pct >= _SAVINGS_NORM else "ниже нормы 20%"
        lines.append(f"Накопления: {health.savings_pct}% дохода ({verdict}).")
    cut = await _top_wants(family_id, repo=repo, now=now, limit=1)
    if cut and (health.savings_pct is None or health.savings_pct < _SAVINGS_NORM):
        cat, amount = cut[0]
        lines.append(f"Можно урезать «{cat.value}» — {_money(amount)} в этом месяце.")
    if progress is not None:
        monthly = progress.monthly_needed(now)
        goal_line = f"Цель: накоплено {progress.pct}%."
        if monthly is not None and not progress.reached:
            goal_line += f" Нужно {_money(monthly)}/мес."
        lines.append(goal_line)

    if len(lines) == 1:
        return None
    return "\n".join(lines)


async def goal_status_text(family_id: uuid.UUID, *, now: datetime | None = None) -> str:
    """Format the family's savings goal + progress for the ``/goal`` command."""
    now = now or datetime.now(tz=_MOSCOW)
    repo = MCPLedgerReader()
    goal = await repo.get_savings_goal(family_id=family_id)
    if goal is None:
        return (
            "Цель накопления не задана.\n"
            "Поставь так: <code>/goal 200000 до 31.12.2026</code> "
            "(дата необязательна)."
        )
    progress = await _goal_progress(goal, repo=repo, now=now)
    head = f"🎯 <b>Цель:</b> {_money(goal.target_amount)}"
    if goal.target_date is not None:
        head += f" к {goal.target_date.strftime('%d.%m.%Y')}"
    lines = [
        head,
        f"Накоплено: {_money(progress.saved_so_far)} ({progress.pct}%)",
        f"Осталось: {_money(progress.remaining)}",
    ]
    if progress.reached:
        lines.append("✅ Цель достигнута!")
    else:
        monthly = progress.monthly_needed(now)
        if monthly is not None:
            lines.append(f"Откладывай {_money(monthly)}/мес, чтобы успеть.")
        on_track = progress.on_track(now)
        if on_track is True:
            lines.append("🟢 Идёшь по графику.")
        elif on_track is False:
            lines.append("🔴 Отстаёшь от графика.")
    return "\n".join(lines)


# ── Money formatting ──────────────────────────────────────────────────────────


def _money(value: Decimal) -> str:
    """Format Decimal as ``1 234 ₽`` (space thousands sep)."""
    int_part = int(value)
    return f"{int_part:,}".replace(",", " ") + " ₽"
