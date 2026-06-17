"""Weekly digest builder.

Composes a short Russian summary of the past week's spending: total,
top categories, top merchants and any subscription anomalies. Delivered
either by the APScheduler job (Sunday 19:00 МСК) or on demand via the
``/digest`` Telegram command.

The numbers come from SQL aggregates; the narrative wrapper is a single
LLM call (worker tier) so the digest reads as a human summary rather
than a table.
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from family_finance.agents._messages import message_text
from family_finance.agents.advisor import build_advice_block
from family_finance.agents.subscriptions import detect_alerts
from family_finance.domain import Category
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository

logger = structlog.get_logger()

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

_DIGEST_SYSTEM = """\
Ты — финансовый помощник семьи. Кратко резюмируй траты за прошедшую неделю
по-русски: 3-5 предложений, дружелюбный тон, конкретные цифры с пробелами
между тысячами (например «12 300 ₽»).

Структура: общая сумма → 1-2 главные категории → если есть алерты — упомяни.
Не выдумывай — используй только переданные факты.
"""


def _trace_config(family_id: uuid.UUID) -> RunnableConfig:
    """LangFuse callback для дайджеста: вызывается в обход графа (/digest +
    плановая рассылка), поэтому callback нужно прикрепить вручную, иначе trace
    не появится. Сессия — на семью, своя от чат-тредов ``tg:<chat_id>``."""
    return {
        "callbacks": cast("list[BaseCallbackHandler]", [make_callback_handler()]),
        "metadata": {
            "langfuse_user_id": str(family_id),
            "langfuse_session_id": f"digest:{family_id}",
            "langfuse_tags": ["digest", "phase2"],
            "langfuse_trace_name": "weekly-digest",
        },
    }


def _week_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return [start, end) for the most recently completed Mon..Sun week."""
    now = now or datetime.now(_MOSCOW)
    # Week starts Monday in Russia. We summarise the week that just ended:
    # end = the Monday boundary closing the current week (today 00:00 when now
    # is already Monday — без `or 7`, иначе в пн окно прыгало на неделю вперёд,
    # QA-12), start = end - 7 days.
    end_local = (now + timedelta(days=(7 - now.weekday()) % 7)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_local = end_local - timedelta(days=7)
    return start_local, end_local


async def build_digest(family_id: uuid.UUID, *, now: datetime | None = None) -> str | None:
    """Compose the weekly digest text for *family_id*.

    Returns ``None`` if there were no expenses that week — the scheduler
    silently skips sending in that case so we don't spam quiet families.
    """
    start, end = _week_window(now)
    repo = PostgresTransactionRepository()
    breakdown = await repo.category_breakdown(family_id=family_id, start=start, end=end)

    if not breakdown:
        return None

    total = sum((amount for _, amount, _ in breakdown), Decimal("0"))
    top_cats = breakdown[:3]
    alerts: list[str] = []
    try:
        alerts = await detect_alerts(family_id=family_id)
    except Exception:
        logger.exception("digest_alerts_failed")

    facts = _format_facts(start, end, total, top_cats, alerts)
    try:
        model = get_chat_model(tier="worker")
        response = await model.ainvoke(
            [SystemMessage(content=_DIGEST_SYSTEM), HumanMessage(content=facts)],
            config=_trace_config(family_id),
        )
        narrative = message_text(response).strip()
    except Exception:
        logger.exception("digest_llm_failed")
        narrative = _fallback_narrative(total, top_cats)

    body = f"📊 <b>Итоги недели</b>\n\n{narrative}"
    if alerts:
        body += "\n\n" + "\n".join(alerts)

    try:
        advice = await build_advice_block(family_id, now=now)
    except Exception:
        logger.exception("digest_advice_failed")
        advice = None
    if advice:
        body += "\n\n" + advice
    return body


def _format_facts(
    start: datetime,
    end: datetime,
    total: Decimal,
    top_cats: list[tuple[Category, Decimal, int]],
    alerts: list[str],
) -> str:
    cat_lines = "\n".join(
        f"  - {cat.value}: {_money(amount)} ({n} операций)" for cat, amount, n in top_cats
    )
    alert_line = f"\nАлерты подписок: {len(alerts)}" if alerts else "\nАлертов подписок нет."
    return (
        f"Период: {start.strftime('%d.%m')} - {(end - timedelta(days=1)).strftime('%d.%m.%Y')}\n"
        f"Всего расходов: {_money(total)}\n"
        f"Главные категории:\n{cat_lines}"
        f"{alert_line}"
    )


def _fallback_narrative(total: Decimal, top_cats: list[tuple[Category, Decimal, int]]) -> str:
    """Used if LLM is unavailable — still useful, just less natural."""
    parts = [f"За прошедшую неделю расходов на {_money(total)}."]
    if top_cats:
        top = top_cats[0]
        parts.append(f"Больше всего на «{top[0].value}» — {_money(top[1])}.")
    return " ".join(parts)


def _money(value: Decimal) -> str:
    int_part = int(value)
    return f"{int_part:,}".replace(",", " ") + " ₽"
