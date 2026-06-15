"""SubscriptionAgent: detect and report recurring expenses.

Triggered by:
  * explicit /subscriptions Telegram command (handler invokes the graph
    with a synthetic ``"подписки"`` message), or
  * a natural-language question like "какие у меня подписки?".

Deterministic detection lives in the repo (SQL aggregate). The agent's job
is to format the result list as a short Russian summary.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import structlog
from langchain_core.messages import AIMessage

from family_finance.agents.state import FinanceState, SectionResult
from family_finance.domain import Subscription
from family_finance.infrastructure.mcp import MCPLedgerReader

logger = structlog.get_logger()


_TRIGGER_TOKENS = (
    "подписк",
    "регулярн",
    "повторяющ",
    "ежемесячн",
    "subscriptions",
)


def is_subscriptions_question(text: str) -> bool:
    """Lightweight intent detector for SubscriptionAgent."""
    normalized = text.lower().replace("ё", "е")
    return any(token in normalized for token in _TRIGGER_TOKENS)


async def subscriptions_node(state: FinanceState) -> dict[str, object]:
    """Find recurring expenses and report them to the user."""
    family_id = uuid.UUID(state["family_id"])
    repo = MCPLedgerReader()
    subs = await repo.detect_recurring(family_id=family_id)

    if not subs:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не нашёл регулярных трат за последний год. "
                        "Загрузи побольше истории — тогда смогу детектировать подписки."
                    )
                )
            ],
            "current_intent": "idle",
        }

    return {
        "messages": [AIMessage(content=format_subscriptions(subs))],
        "current_intent": "idle",
    }


async def build_subscriptions_section(family_id: uuid.UUID) -> SectionResult:
    """Recurring-payments list — one orchestrator-worker section (ADR 0008).

    Детектор подписок работает по всей истории (lookback ~год), периодом запроса
    не ограничивается — поэтому period сюда не передаётся.
    """
    repo = MCPLedgerReader()
    subs = await repo.detect_recurring(family_id=family_id)
    body = format_subscriptions(subs) if subs else "📅 Регулярных трат за последний год не нашёл."
    return {"kind": "subscriptions", "order": 3, "title": "Подписки", "body": body}


_DAYS_PER_MONTH = Decimal("30")


def _monthly_equivalent(sub: Subscription) -> Decimal:
    """Привести трату к месячному эквиваленту: недельная ×~4, годовая ÷12.

    Суммировать `average_amount` напрямую нельзя — у подписок разный cadence_days,
    тогда «/мес» был бы арифметически неверным.
    """
    return sub.average_amount * _DAYS_PER_MONTH / Decimal(sub.cadence_days)


def format_subscriptions(subs: list[Subscription]) -> str:
    """Render the detected subscriptions list as a Russian Telegram message."""
    total_monthly = sum((_monthly_equivalent(s) for s in subs), Decimal("0"))
    lines = [f"📅 Нашёл регулярных трат: {len(subs)} (≈ {_money(total_monthly)} / мес)", ""]
    for sub in subs:
        amount = _money(sub.last_amount)
        avg_part = ""
        if sub.last_amount != sub.average_amount:
            delta_pct = _pct_change(sub.last_amount, sub.average_amount)
            avg_part = f" (обычно {_money(sub.average_amount)}, {delta_pct:+.0f}%)"
        last_seen = sub.last_seen.strftime("%d.%m.%Y")
        lines.append(
            f"• <b>{sub.merchant}</b> — {amount}{avg_part}, "
            f"раз в ~{sub.cadence_days} дн., последнее {last_seen}"
        )
    return "\n".join(lines)


def _money(value: Decimal) -> str:
    """Format Decimal as `1 234 ₽`."""
    int_part = int(value)
    formatted = f"{int_part:,}".replace(",", " ")
    return f"{formatted} ₽"


def _pct_change(current: Decimal, baseline: Decimal) -> float:
    if baseline == Decimal("0"):
        return 0.0
    return float((current - baseline) / baseline * 100)


# ── Post-import alerts ────────────────────────────────────────────────────────

# Subscription needs at least this many prior occurrences to be considered a
# baseline against which a new payment can be compared. Lower → noisier
# (alerts fire on first imports), higher → quieter but slower to flag changes.
_ALERT_MIN_OCCURRENCES = 4

# Trigger an alert when the latest payment differs from the historical average
# by at least this fraction (15% by default).
_ALERT_AMOUNT_DELTA = Decimal("0.15")

# Не заваливать пользователя — отдаём не больше этого числа алертов (самые
# крупные по модулю изменения), остальное шумит.
_ALERT_MAX_COUNT = 5


async def detect_alerts(family_id: uuid.UUID) -> list[str]:
    """Return short, human-readable alert lines for subscriptions whose last
    payment diverged from their average.

    Called after a fresh import has been categorised — we re-run the
    recurring-payment detector and surface anything that looks like a price
    change. Cheap (one SQL aggregate, no LLM).
    """
    repo = MCPLedgerReader()
    subs = await repo.detect_recurring(family_id=family_id)
    scored: list[tuple[Decimal, str]] = []
    for sub in subs:
        if sub.occurrences < _ALERT_MIN_OCCURRENCES:
            continue
        if sub.average_amount == Decimal("0"):
            continue
        delta = (sub.last_amount - sub.average_amount) / sub.average_amount
        if abs(delta) < _ALERT_AMOUNT_DELTA:
            continue
        direction = "подорожала" if delta > 0 else "подешевела"
        line = (
            f"⚠️ Подписка <b>{sub.merchant}</b> {direction}: "
            f"{_money(sub.last_amount)} вместо обычных {_money(sub.average_amount)} "
            f"({_pct_change(sub.last_amount, sub.average_amount):+.0f}%)"
        )
        scored.append((abs(delta), line))
    # Самые крупные изменения первыми, остальное обрезаем — иначе шум.
    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in scored[:_ALERT_MAX_COUNT]]
