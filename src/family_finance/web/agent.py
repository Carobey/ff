"""Web adapter for asking the existing finance agent graph."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.agents import build_supervisor_graph
from family_finance.agents._messages import message_text
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.web.dashboard import DashboardRepository

_TELEGRAM_HTML_RE = re.compile(r"</?(?:b|i|u|s|code|pre)>")


@dataclass(frozen=True)
class AgentAnswer:
    answer: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_graph: CompiledStateGraph[Any, Any, Any, Any] | None = None


def _get_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    global _graph
    if _graph is None:
        _graph = build_supervisor_graph()
    return _graph


async def ask_agent(
    *,
    family_id: uuid.UUID,
    question: str,
    repo: DashboardRepository | None = None,
) -> AgentAnswer:
    """Ask the existing supervisor graph a web-originated question."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Пустой вопрос")

    dashboard_repo: DashboardRepository = repo or PostgresTransactionRepository()
    member_id = await dashboard_repo.get_primary_member_id(family_id=family_id)
    if member_id is None:
        raise ValueError("У выбранной семьи нет участника")

    thread_id = f"web:{family_id}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": cast("list[BaseCallbackHandler]", [make_callback_handler()]),
        "metadata": {
            "web_family_id": str(family_id),
            "langfuse_user_id": str(family_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["web"],
            "langfuse_trace_name": "web_agent",
        },
    }
    result = await _get_graph().ainvoke(
        {
            "messages": [HumanMessage(content=clean_question)],
            "family_id": str(family_id),
            "member_id": str(member_id),
            "telegram_user_id": 0,
            "telegram_chat_id": 0,
        },
        config=config,
    )
    interrupts = result.get("__interrupt__")
    if interrupts:
        return AgentAnswer(answer="Агенту нужны дополнительные данные для продолжения.")
    messages = result.get("messages", [])
    if not messages:
        return AgentAnswer(answer="Агент не вернул ответ.")
    return AgentAnswer(answer=_web_text(message_text(messages[-1])))


def _web_text(text: str) -> str:
    """Strip lightweight Telegram HTML tags before returning text to the browser."""
    return _TELEGRAM_HTML_RE.sub("", text)
