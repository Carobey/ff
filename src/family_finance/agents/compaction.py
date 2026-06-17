"""Compaction node: keep long Telegram threads from bloating the checkpoint.

Один тред чата живёт в одном PostgresSaver-checkpoint и копит ``messages``
бесконечно. На длинных диалогах это раздувает state и трейсы LangFuse. Нода
сворачивает старую часть истории в одну краткую сводку (``SystemMessage``),
оставляя несколько последних сообщений дословно. Запускается ПЕРЕД supervisor;
ниже порога — no-op без LLM (ADR 0008, фаза полировки).

Паттерн LangGraph: ``RemoveMessage(REMOVE_ALL_MESSAGES)`` очищает список, затем в
том же апдейте докладываем ``[summary, *recent]`` — порядок гарантирован, сводка
впереди. Прошлая сводка попадает в ``older`` и сворачивается в новую (rolling).
"""

from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from family_finance.agents.state import FinanceState
from family_finance.infrastructure.llm import get_chat_model

logger = logging.getLogger(__name__)

# Свернуть, когда сообщений больше этого порога; оставить столько последних.
_COMPACT_AFTER = 20
_KEEP_RECENT = 8

_SUMMARY_SYSTEM = """\
Ты сжимаешь историю диалога финансового помощника в краткую сводку для памяти.
Сохрани факты, важные для будущих ответов: о чём пользователь спрашивал, его
предпочтения, незакрытые темы, упомянутые суммы/периоды/категории. Без воды,
по-русски, до 8 строк. Не выдумывай — только то, что было в диалоге.
"""


def _render(messages: list[BaseMessage]) -> str:
    """Flatten messages into a plain role-tagged transcript for the summarizer."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "Пользователь"
        elif isinstance(m, AIMessage):
            role = "Помощник"
        else:
            role = "Система"
        lines.append(f"{role}: {str(m.content).strip()}")
    return "\n".join(lines)


async def compact_node(state: FinanceState) -> dict[str, object]:
    """Summarize the older half of a long thread; keep the recent tail verbatim."""
    messages = state.get("messages", [])
    if len(messages) <= _COMPACT_AFTER:
        return {}

    # Хвост не должен начинаться с осиротевшего ToolMessage: его инициирующий
    # AIMessage (с tool_calls) ушёл бы в свёрнутую часть, и провайдер отклонит
    # tool-result без предшествующего tool-call. Сдвигаем границу назад, пока
    # хвост не начнётся с не-ToolMessage — так пара tool-call/result остаётся
    # целой. Latent-гард: сегодня ReAct-циклы (advisor/coach) кладут в state
    # только финальный ответ, ToolMessage в граф не попадают — страховка на
    # случай, если начнут (PR-10).
    split = len(messages) - _KEEP_RECENT
    while split > 0 and isinstance(messages[split], ToolMessage):
        split -= 1
    older = messages[:split]
    recent = messages[split:]

    model = get_chat_model(tier="worker")
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=_SUMMARY_SYSTEM),
                HumanMessage(content=_render(older)),
            ],
        )
    except Exception:
        # Транзиентный сбой summarizer'а не должен валить весь ход: на длинном
        # треде compact_node на горячем пути КАЖДОГО сообщения. No-op — тред не
        # свернётся в этот раз, supervisor отработает как обычно, ретрай на след.
        # ходе (ср. _plan_query в supervisor.py).
        logger.exception("thread_compaction_failed from=%d", len(messages))
        return {}
    summary = SystemMessage(content=f"[Сводка предыдущего диалога]\n{response.content}")
    logger.info("thread_compacted from=%d kept=%d", len(messages), len(recent))

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary, *recent]}
