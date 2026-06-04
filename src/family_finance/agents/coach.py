"""CoachAgent: behavioural finance queries via Graphiti episodic memory (P2-19).

Handles questions like:
  "Когда я последний раз так часто заказывал доставку?"
  "Я обычно столько трачу на одежду или это аномалия?"
  "В какой месяц этого года я тратил меньше всего?"

Flow:
  1. search_episodes(user_query, group_id=family_id) → EntityEdge[]
  2. Format edges as context facts
  3. LLM (worker) generates a 2-4 sentence narrative from those facts
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from family_finance.agents._messages import message_text
from family_finance.agents.state import FinanceState
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.memory.graphiti_client import search_episodes

logger = structlog.get_logger()


@runtime_checkable
class _GraphitiEdge(Protocol):
    """Structural shape of a Graphiti EntityEdge we care about.

    We don't import ``graphiti_core.edges.EntityEdge`` directly because the
    coach module shouldn't depend on Graphiti's internal class layout — any
    object with the right attributes works (handy for tests too).
    """

    fact: str | None
    name: str | None


_COACH_SYSTEM = """\
Ты — финансовый коуч семьи. Анализируй паттерны расходов и отвечай по-русски.
Правила:
- Отвечай 2-4 предложениями, конкретно и с цифрами если они есть
- Используй ТОЛЬКО факты из предоставленного контекста, не придумывай
- Если контекста недостаточно для ответа — честно скажи об этом
- Не давай финансовых советов — только наблюдения из истории трат
"""

_NO_DATA_REPLY = (
    "Недостаточно данных в истории для ответа на этот вопрос. "
    "Загрузи выписку банка или больше чеков — тогда смогу анализировать паттерны."
)


async def coach_node(state: FinanceState) -> dict[str, object]:
    """Answer a behavioural finance question using Graphiti episodic memory."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else ""
    family_id = str(uuid.UUID(state["family_id"]))

    # 1. Semantic search in Graphiti knowledge graph
    edges = await search_episodes(
        query=user_text,
        group_id=family_id,
        num_results=15,
    )

    if not edges:
        return {
            "messages": [AIMessage(content=_NO_DATA_REPLY)],
            "current_intent": "idle",
        }

    # 2. Format edges as context facts for LLM
    context_lines: list[str] = []
    for edge in edges:
        fact = _edge_to_fact(edge)
        if fact:
            context_lines.append(f"• {fact}")

    if not context_lines:
        return {
            "messages": [AIMessage(content=_NO_DATA_REPLY)],
            "current_intent": "idle",
        }

    context = "\n".join(context_lines[:15])
    user_prompt = f"Вопрос пользователя: «{user_text}»\n\nФакты из истории трат:\n{context}"

    # 3. LLM narrative
    try:
        model = get_chat_model(tier="worker")
        response = await model.ainvoke(
            [SystemMessage(content=_COACH_SYSTEM), HumanMessage(content=user_prompt)],
        )
        reply = message_text(response)
    except Exception:
        logger.exception("coach_node: llm failed")
        # Fallback: return raw facts
        reply = "Нашёл в истории:\n" + context

    return {
        "messages": [AIMessage(content=reply)],
        "current_intent": "idle",
    }


def _edge_to_fact(edge: object) -> str | None:
    """Extract a human-readable fact string from a Graphiti EntityEdge."""
    if not isinstance(edge, _GraphitiEdge):
        return None
    if edge.fact and isinstance(edge.fact, str):
        return edge.fact.strip()
    if edge.name and isinstance(edge.name, str):
        return edge.name.strip()
    return None
