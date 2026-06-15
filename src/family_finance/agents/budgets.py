"""BudgetsAgent: monthly per-category budgets + post-import alerts.

Components:
  * :func:`current_moscow_month` — month boundary helper.
  * :func:`format_budgets` — render the status list for ``/budgets``.
  * :func:`detect_budget_alerts` — produce alert lines for any budget
    that has crossed the 80% or 100% threshold this month. Called by
    the categorizer after a fresh import.
  * :func:`budgets_node` — LangGraph node that powers ``/budgets``.
  * :func:`parse_budget_category` — accept a Category enum value OR a
    Russian keyword (e.g. ``"продукты" → FOOD_GROCERIES``).
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal

import structlog
from langchain_core.messages import AIMessage

from family_finance.agents.state import FinanceState, SectionResult
from family_finance.domain import BudgetStatus, Category
from family_finance.infrastructure.mcp import MCPLedgerReader

logger = structlog.get_logger()

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

# Budget alert thresholds.
_WARN_PCT = 80  # yellow
_OVER_PCT = 100  # red


# ── Month-window helper ───────────────────────────────────────────────────────


def current_moscow_month(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return ``[start, end)`` for the current calendar month in Europe/Moscow."""
    now = now or datetime.now(_MOSCOW)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Roll the day forward into next month then truncate to day=1.
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


# ── Display ───────────────────────────────────────────────────────────────────


_TRIGGER_TOKENS = ("бюджет", "лимит", "budget")


def is_budgets_question(text: str) -> bool:
    normalized = text.lower().replace("ё", "е")
    return any(tok in normalized for tok in _TRIGGER_TOKENS)


def format_budgets(statuses: list[BudgetStatus]) -> str:
    """Render the budgets list for ``/budgets``."""
    if not statuses:
        return "Бюджеты ещё не настроены.\nПоставь их так: <code>/budget продукты 30000</code>"
    lines = ["💰 <b>Бюджеты на этот месяц</b>", ""]
    for s in statuses:
        icon = "🔴" if s.pct >= _OVER_PCT else ("🟡" if s.pct >= _WARN_PCT else "🟢")
        lines.append(
            f"{icon} <b>{s.budget.category.value}</b>: "
            f"{_money(s.spent_this_month)} / {_money(s.budget.monthly_limit)} ({s.pct}%)"
        )
    return "\n".join(lines)


# ── Alerts ────────────────────────────────────────────────────────────────────


async def detect_budget_alerts(family_id: uuid.UUID) -> list[str]:
    """Return alert lines for budgets at >=80% or over limit this month."""
    month_start, month_end = current_moscow_month()
    repo = MCPLedgerReader()
    try:
        statuses = await repo.get_budget_status(
            family_id=family_id,
            month_start=month_start,
            month_end=month_end,
        )
    except Exception:
        logger.exception("budget_status_failed")
        return []

    alerts: list[str] = []
    for s in statuses:
        if s.pct >= _OVER_PCT:
            alerts.append(
                f"🔴 Бюджет на «{s.budget.category.value}» превышен: "
                f"{_money(s.spent_this_month)} из {_money(s.budget.monthly_limit)} ({s.pct}%)"
            )
        elif s.pct >= _WARN_PCT:
            alerts.append(
                f"🟡 Бюджет на «{s.budget.category.value}»: "
                f"{_money(s.spent_this_month)} из {_money(s.budget.monthly_limit)} ({s.pct}%)"
            )
    return alerts


# ── LangGraph node ────────────────────────────────────────────────────────────


async def budgets_node(state: FinanceState) -> dict[str, object]:
    """Show the family's budgets and current month's status."""
    family_id = uuid.UUID(state["family_id"])
    month_start, month_end = current_moscow_month()
    repo = MCPLedgerReader()
    statuses = await repo.get_budget_status(
        family_id=family_id,
        month_start=month_start,
        month_end=month_end,
    )
    return {
        "messages": [AIMessage(content=format_budgets(statuses))],
        "current_intent": "idle",
    }


# ── Orchestrator section (ADR 0008) ───────────────────────────────────────────


async def build_budgets_section(
    family_id: uuid.UUID,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> SectionResult:
    """Budget status for the given period — one orchestrator-worker section.

    Period передаётся явно из планировщика; если он не распарсен («за все
    время») — падаем на текущий московский месяц, как в одиночной ноде.
    """
    if start is None or end is None:
        start, end = current_moscow_month()
    repo = MCPLedgerReader()
    statuses = await repo.get_budget_status(
        family_id=family_id,
        month_start=start,
        month_end=end,
    )
    return {
        "kind": "budgets",
        "order": 2,
        "title": "Бюджеты",
        "body": format_budgets(statuses),
    }


# ── Category parsing ──────────────────────────────────────────────────────────


# Намеренно отдельная таблица (см. подробный комментарий в ``ledger_terms``).
# Здесь: бюджет-ввод → ОДНА категория (1→1, без направления — оно выводится из
# категории). «магазин» здесь = продуктовый бюджет (FOOD_GROCERIES); в
# ``clarifications`` тот же токен ведёт в SHOPPING_GENERIC — слить нельзя.
_RU_CATEGORY_ALIASES: dict[tuple[str, ...], Category] = {
    ("продукт", "еда", "магазин", "супермаркет"): Category.FOOD_GROCERIES,
    ("ресторан", "кафе", "столовая"): Category.FOOD_RESTAURANT,
    ("доставк",): Category.FOOD_DELIVERY,
    ("бензин", "топлив", "азс"): Category.TRANSPORT_FUEL,
    ("такси",): Category.TRANSPORT_TAXI,
    ("метро", "автобус", "транспорт"): Category.TRANSPORT_PUBLIC,
    ("одежд",): Category.SHOPPING_CLOTHES,
    ("покупк", "маркетплейс"): Category.SHOPPING_GENERIC,
    ("жкх", "коммун", "квартир"): Category.HOME_UTILITIES,
    ("ремонт",): Category.HOME_REPAIR,
    ("аптек", "лекарств"): Category.HEALTH_PHARMACY,
    ("врач", "клиник", "красот", "здоровь"): Category.HEALTH_GENERIC,
    ("подпис",): Category.ENTERTAINMENT_SUBS,
    ("питом", "ветеринар", "зоо"): Category.PETS,
    ("детск", "ребен", "ребён"): Category.KIDS_TOYS,
}


def parse_budget_category(raw: str) -> Category | None:
    """Map free-form user input to a :class:`Category`.

    Accepts dot-notation enum values (``"food.groceries"``) or Russian
    keywords (``"продукты"``, ``"ЖКХ"``). Returns ``None`` when there's
    no match — caller asks the user to clarify.
    """
    text = raw.strip()
    try:
        return Category(text)
    except ValueError:
        pass

    normalized = text.lower().replace("ё", "е")
    for tokens, category in _RU_CATEGORY_ALIASES.items():
        if any(tok in normalized for tok in tokens):
            return category
    return None


# ── Money formatting ──────────────────────────────────────────────────────────


def _money(value: Decimal) -> str:
    """Format Decimal as ``1 234 ₽`` (rouble, space thousands sep)."""
    int_part = int(value)
    return f"{int_part:,}".replace(",", " ") + " ₽"
