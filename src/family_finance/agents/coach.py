"""CoachAgent: behavioural finance queries (QA-01, was P2-19).

Handles open-ended questions about spending behaviour:
  "Как часто я заказываю доставку?"
  "Когда я последний раз так тратил?"
  "А так ли это для меня типично?"

Approach — ReAct over MCP tools, mirroring ``advisor_node``: the LLM decides
*what* to look up and narrates the result, but every number comes from a
Python tool (MCP aggregate / transaction list), never from the model. Graphiti
episodic memory is exposed as one optional tool (``recall_episodes``) for
qualitative recall, so the diploma's episodic-memory story (P2-18/19) is kept
without being the only data source — which was the root cause of the old
"нет данных" answers (it held aggregate import episodes, no dated facts).

``family_id`` is baked into the tool closures — the LLM never sees the UUID and
cannot query another family's data (security layer, like advisor).
"""

from __future__ import annotations

import uuid
import zoneinfo
from datetime import datetime
from decimal import Decimal
from typing import Literal

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

from family_finance.agents._messages import message_text, recent_dialog
from family_finance.agents.ledger import parse_period
from family_finance.agents.ledger_terms import CATEGORY_LABELS
from family_finance.agents.state import FinanceState
from family_finance.domain import Category, Direction
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.mcp import call_finance_tool
from family_finance.infrastructure.memory.graphiti_client import search_episodes

logger = structlog.get_logger()

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

_COACH_REACT_SYSTEM = """\
Ты — финансовый коуч семьи. Отвечаешь на поведенческие вопросы о тратах
(как часто, когда последний раз, типично ли, есть ли всплеск) по-русски.

У тебя есть инструменты, которые отдают УЖЕ ПОСЧИТАННЫЕ Python-ом цифры по
этой семье:
- spending_breakdown — суммы и КОЛИЧЕСТВО операций в разбивке по категориям,
  продавцам, месяцам или неделям за период (count = как часто).
- recent_transactions — последние операции с датами (для «когда последний раз»).
- recall_episodes — качественные заметки из истории (эпизодическая память).

ПРАВИЛА:
- Сначала вызови нужные инструменты, чтобы узнать реальные цифры/даты. Используй
  ТОЛЬКО эти данные — НИКОГДА не выдумывай суммы, количества и даты.
- Для «как часто» смотри количество операций (count) за период.
- Для «когда последний раз» бери дату из recent_transactions.
- Дёргай ровно те инструменты, что нужны под вопрос; не вызывай лишние.
- Итог — 2-4 предложения, конкретно, суммы с пробелами между тысячами («12 300 ₽»).
- Только наблюдения из истории, без финансовых советов.
- Если данных нет — честно скажи и предложи загрузить выписку.
"""

_NO_DATA_REPLY = (
    "Недостаточно данных в истории для ответа на этот вопрос. "
    "Загрузи выписку банка или больше чеков — тогда смогу анализировать паттерны."
)

_DIM_TITLE: dict[str, str] = {
    "category": "по категориям",
    "merchant": "по продавцам",
    "month": "по месяцам",
    "week": "по неделям",
}


async def coach_node(state: FinanceState) -> dict[str, object]:
    """Answer a behavioural finance question via a ReAct loop over MCP tools.

    Числа считает Python (MCP-инструменты), LLM лишь решает, что спросить, и
    оборачивает результат в наблюдение. Маскирование PII сохраняется — ReAct
    ходит через ``MaskingChatModel``.
    """
    family_id = uuid.UUID(state["family_id"])

    # Дешёвый гейт без LLM: совсем пустая история → честный ответ, не гоняем
    # ReAct-цикл впустую.
    if not await _has_transactions(family_id):
        return {"messages": [AIMessage(content=_NO_DATA_REPLY)], "current_intent": "idle"}

    # История диалога (хвост), чтобы уточняющие вопросы держали антецедент.
    dialog = recent_dialog(state.get("messages", []))
    tools = _build_coach_tools(family_id)
    try:
        agent = create_react_agent(
            get_chat_model(tier="worker"),
            tools,
            prompt=_COACH_REACT_SYSTEM,
        )
        config: RunnableConfig = {"recursion_limit": 10}
        result = await agent.ainvoke({"messages": dialog}, config=config)
        reply = message_text(result["messages"][-1]).strip()
        if not reply:
            raise ValueError("empty coach reply")
    except Exception:
        logger.exception("coach_react_failed")
        reply = _NO_DATA_REPLY

    return {"messages": [AIMessage(content=reply)], "current_intent": "idle"}


async def _has_transactions(family_id: uuid.UUID) -> bool:
    """True if the family has at least one expense on record (all-time)."""
    rows = await call_finance_tool(
        "query_aggregates",
        {
            "family_id": str(family_id),
            "group_by": "total",
            "directions": [Direction.EXPENSE.value],
        },
    )
    return bool(rows) and int(rows[0].get("count", 0)) > 0


def _build_coach_tools(family_id: uuid.UUID) -> list[BaseTool]:
    """ReAct-инструменты коуча. ``family_id`` зашит в замыкание — LLM никогда не
    видит UUID семьи и не может запросить чужие данные (см. security-слой).

    Все числа приходят из MCP (SQL-агрегаты) и Graphiti — модель только читает.
    """

    @tool
    async def spending_breakdown(
        dimension: Literal["category", "merchant", "month", "week"],
        period_hint: str = "",
    ) -> str:
        """Суммы и количество операций в разбивке за период.

        ``dimension`` — ось разбивки: category/merchant/month/week.
        ``period_hint`` — свободный текст периода («в апреле», «за прошлый месяц»,
        пусто = за всё время). Для «как часто» смотри количество операций.
        """
        start, end, period_label = parse_period(period_hint)
        rows = await call_finance_tool(
            "query_aggregates",
            {
                "family_id": str(family_id),
                "group_by": dimension,
                "directions": [Direction.EXPENSE.value],
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
                "limit": 20,
            },
        )
        if not rows:
            return f"Трат {period_label} не найдено."
        lines = [f"Траты {period_label} — {_DIM_TITLE.get(dimension, dimension)}:"]
        for r in rows:
            label = _bucket_label(dimension, str(r["bucket"]))
            total = Decimal(str(r["total"]))
            count = int(r["count"])
            lines.append(f"• {label}: {_money(total)} ({count} операций)")
        return "\n".join(lines)

    @tool
    async def recent_transactions(period_hint: str = "", limit: int = 15) -> str:
        """Последние операции с датами (новые сверху).

        ``period_hint`` — свободный текст периода (пусто = за всё время).
        Используй для вопросов «когда последний раз …».
        """
        start, end, period_label = parse_period(period_hint)
        rows = await call_finance_tool(
            "list_transactions",
            {
                "family_id": str(family_id),
                "directions": [Direction.EXPENSE.value],
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
                "order_by": "date_desc",
                "limit": max(1, min(limit, 50)),
            },
        )
        if not rows:
            return f"Операций {period_label} не найдено."
        lines = [f"Последние операции {period_label}:"]
        for e in rows:
            occurred = datetime.fromisoformat(str(e["occurred_at"]))
            label = _bucket_label("category", str(e["category"]))
            merchant = str(e["merchant"]) or "—"
            lines.append(
                f"• {occurred.strftime('%d.%m.%Y')} {merchant} ({label}): "
                f"{_money(Decimal(str(e['amount'])))}"
            )
        return "\n".join(lines)

    @tool
    async def recall_episodes(query: str) -> str:
        """Качественные заметки из истории трат (эпизодическая память).

        Поиск по смыслу в графе знаний семьи. Используй для контекста, когда
        нужны не цифры, а наблюдения («был ли всплеск», «что необычного»).
        """
        edges = await search_episodes(query=query, group_id=str(family_id), num_results=10)
        facts = [f"• {fact}" for edge in edges if (fact := _edge_to_fact(edge))]
        if not facts:
            return "В эпизодической памяти ничего по этому запросу не нашлось."
        return "Заметки из истории:\n" + "\n".join(facts)

    return [spending_breakdown, recent_transactions, recall_episodes]


def _bucket_label(dimension: str, bucket: str) -> str:
    """Human label for one aggregation bucket key (category → RU label)."""
    if dimension == "category":
        try:
            return CATEGORY_LABELS.get(Category(bucket), bucket)
        except ValueError:
            return bucket
    return bucket  # merchant / month (YYYY-MM) / week (date) — raw


def _money(value: Decimal) -> str:
    """Format Decimal as ``1 234 ₽`` (space thousands sep)."""
    return f"{int(value):,}".replace(",", " ") + " ₽"


def _edge_to_fact(edge: object) -> str | None:
    """Extract a human-readable fact string from a Graphiti EntityEdge.

    Duck-typed on purpose: we don't import Graphiti's ``EntityEdge`` class, any
    object exposing a ``fact``/``name`` string works (also handy for tests).
    """
    for attr in ("fact", "name"):
        value = getattr(edge, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
