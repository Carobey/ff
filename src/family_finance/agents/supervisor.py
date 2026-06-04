"""
Supervisor — минимальная версия Phase 0.

ЦЕЛЬ: показать что инфра жива.
Принимает сообщение → LLM отвечает → checkpoint сохранён → trace в LangFuse.

Phase 1: добавляются ноды Ingest/Receipt/Categorizer/Ledger и conditional_edges.
Phase 2: добавляется Coach с Graphiti memory.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from family_finance.agents._messages import message_text
from family_finance.agents.advisor import advisor_node, is_advice_question
from family_finance.agents.budgets import budgets_node, is_budgets_question
from family_finance.agents.categorizer import categorizer_node
from family_finance.agents.clarify import clarify_node, has_clarification_answers
from family_finance.agents.coach import coach_node
from family_finance.agents.ingest import ingest_node
from family_finance.agents.ledger import is_ledger_question, ledger_node
from family_finance.agents.receipt import receipt_node
from family_finance.agents.state import FinanceState
from family_finance.agents.subscriptions import is_subscriptions_question, subscriptions_node
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.security import REFUSAL_MESSAGE, check_injection

logger = logging.getLogger(__name__)


_PATTERN_TOKENS = (
    "когда я",
    "когда последний",
    "последний раз",
    "как часто",
    "обычно",
    "аномалия",
    "паттерн",
    "привычк",
    "тенденц",
    "похож",
    "так тратил",
    "так же тратил",
    "в какой месяц",
    "самый дорогой месяц",
    "самый дешёвый",
)


def is_pattern_question(text: str) -> bool:
    """Detect behavioural finance questions for CoachAgent."""
    normalized = text.lower()
    return any(token in normalized for token in _PATTERN_TOKENS)


SUPERVISOR_SYSTEM = """Ты — финансовый помощник семьи Юри.

Помогаешь анализировать расходы, отвечаешь на вопросы о тратах по истории транзакций.
Отвечай кратко и дружелюбно (1-3 предложения), по-русски.

Если не понимаешь запрос — попроси уточнить.
"""


async def supervisor_node(state: FinanceState) -> dict[str, object]:
    """Route to a specialist or answer directly via LLM.

    Routing-only branches do NOT emit a placeholder AIMessage: the specialist
    is responsible for the single user-visible reply. Otherwise the user would
    see two messages per request ("получил..." + actual result).
    """
    if state.get("pending_photo"):
        return {"current_intent": "upload_photo"}
    if state.get("pending_csv") or state.get("pending_pdf"):
        return {"current_intent": "upload_csv"}

    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else "Привет"

    if (await check_injection(user_text)).blocked:
        return {
            "messages": [AIMessage(content=REFUSAL_MESSAGE)],
            "current_intent": "idle",
        }

    if has_clarification_answers(state, user_text):
        return {"current_intent": "clarify"}
    if is_ledger_question(user_text):
        return {"current_intent": "query"}
    if is_pattern_question(user_text):
        return {"current_intent": "pattern"}
    if is_subscriptions_question(user_text):
        return {"current_intent": "subscriptions"}
    if is_budgets_question(user_text):
        return {"current_intent": "budgets"}
    if is_advice_question(user_text):
        return {"current_intent": "advice"}

    # No specialist matched — LLM small-talk fallback.
    model = get_chat_model(tier="supervisor")
    response = await model.ainvoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM), HumanMessage(content=user_text)],
    )

    return {
        "messages": [AIMessage(content=message_text(response))],
        "current_intent": "idle",
    }


def route_after_supervisor(state: FinanceState) -> str:
    """Route to specialist nodes using explicit Python logic.

    Returns a node name (or the literal ``"end"`` sentinel mapped to ``END``
    by ``add_conditional_edges``). Using a string instead of the ``END``
    constant keeps the path map's keys homogeneous so mypy can infer
    ``dict[str, str]`` cleanly.
    """
    if state.get("pending_photo"):
        return "receipt"
    if state.get("pending_csv") or state.get("pending_pdf"):
        return "ingest"
    if state.get("current_intent") == "clarify":
        return "clarify"
    if state.get("current_intent") == "query":
        return "ledger"
    if state.get("current_intent") == "pattern":
        return "coach"
    if state.get("current_intent") == "subscriptions":
        return "subscriptions"
    if state.get("current_intent") == "budgets":
        return "budgets"
    if state.get("current_intent") == "advice":
        return "advisor"
    return "end"


def route_after_ingest(state: FinanceState) -> str:
    """Second branch point: categorize only when ingest produced new rows.

    A parser error or an all-duplicates import sets ``ingest_ok=False`` and
    short-circuits to END (ingest already emitted its own message), so the
    categorizer never runs on an empty batch.
    """
    return "categorizer" if state.get("ingest_ok") else "end"


def build_supervisor_graph(
    checkpointer: AsyncPostgresSaver | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Build LangGraph с persistence.

    В рантайме бот всегда передаёт реальный ``AsyncPostgresSaver``.
    ``checkpointer=None`` нужен только офлайн-утилитам (``just printgraph``):
    топология графа для отрисовки не зависит от persistence.
    """
    builder = StateGraph(FinanceState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("ingest", ingest_node)
    builder.add_node("categorizer", categorizer_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("ledger", ledger_node)
    builder.add_node("receipt", receipt_node)
    builder.add_node("coach", coach_node)
    builder.add_node("subscriptions", subscriptions_node)
    builder.add_node("budgets", budgets_node)
    builder.add_node("advisor", advisor_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "ingest": "ingest",
            "clarify": "clarify",
            "ledger": "ledger",
            "receipt": "receipt",
            "coach": "coach",
            "subscriptions": "subscriptions",
            "budgets": "budgets",
            "advisor": "advisor",
            "end": END,
        },
    )
    # ingest → categorizer | END — 2nd branch point: skip categorizer on empty/failed import
    builder.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {"categorizer": "categorizer", "end": END},
    )
    builder.add_edge("categorizer", END)
    builder.add_edge("clarify", END)
    builder.add_edge("ledger", END)
    builder.add_edge("receipt", END)
    builder.add_edge("coach", END)
    builder.add_edge("subscriptions", END)
    builder.add_edge("budgets", END)
    builder.add_edge("advisor", END)

    return builder.compile(checkpointer=checkpointer)
