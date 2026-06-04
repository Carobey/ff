"""Inline-button handler for clarification questions.

Когда ingest+categorizer возвращает open_questions,
documents.py отправляет сообщения с inline-кнопками.
Клик по кнопке → callback_query → synthezises "N category" text → обычный clarify flow.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.settings import Settings

logger = logging.getLogger(__name__)
router = Router(name="clarify_buttons")

# Callback data format: "clarify:{question_id}:{category_value}"
_PREFIX = "clarify"


def make_callback_data(question_id: int, category_value: str) -> str:
    return f"{_PREFIX}:{question_id}:{category_value}"


def parse_callback_data(data: str) -> tuple[int, str] | None:
    """Return (question_id, category_value) or None if not our callback."""
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != _PREFIX:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


@router.callback_query(F.data.startswith(f"{_PREFIX}:"))
async def handle_clarify_button(
    callback: CallbackQuery,
    bot: Bot,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Synthesize text answer and route through clarify node."""
    await callback.answer()  # dismiss the loading spinner

    if callback.data is None or callback.message is None:
        return

    parsed = parse_callback_data(callback.data)
    if parsed is None:
        return

    question_id, category_value = parsed
    # Synthesize the text the user would have typed, e.g. "1 одежда"
    synthetic_text = f"{question_id} {category_value}"

    user_id = callback.from_user.id if callback.from_user else 0
    chat_id = callback.message.chat.id
    thread_id = f"tg:{chat_id}"

    callbacks = cast("list[BaseCallbackHandler]", [make_callback_handler()])
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": {
            "telegram_user_id": str(user_id),
            "telegram_chat_id": str(chat_id),
            "langfuse_user_id": str(user_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["telegram", "phase1", "clarify_button"],
            "langfuse_trace_name": "telegram-clarify-button",
        },
    }

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=synthetic_text)]},
            config=config,
        )
    except Exception:
        logger.exception("clarify button failed")
        await bot.send_message(chat_id, "⚠️ Не смог применить уточнение.")
        return

    reply = result["messages"][-1].content
    await bot.send_message(chat_id, str(reply), parse_mode=None)
