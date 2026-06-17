"""Inline-button handler for the HITL import confirmation (ADR 0009).

ingest_node ставит граф на паузу через ``interrupt()`` перед массовой записью
выписки в БД. documents.py показывает карточку с кнопками «Импортировать / Отмена».
Клик по кнопке → ``Command(resume=bool)`` → граф продолжает запись + категоризацию.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

import structlog
from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from family_finance.bot.handlers.documents import (
    IMPORT_CONFIRM_NO,
    IMPORT_CONFIRM_YES,
    send_import_result,
)
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.settings import Settings

logger = structlog.get_logger()
router = Router(name="import_confirm_buttons")


def _is_allowed(callback: CallbackQuery, settings: Settings) -> bool:
    allowed = settings.allowed_user_ids
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER_IDS пуст — доступ закрыт")
        return False
    user_id = callback.from_user.id if callback.from_user else 0
    return user_id in allowed


@router.callback_query(F.data.in_({IMPORT_CONFIRM_YES, IMPORT_CONFIRM_NO}))
async def handle_import_confirm(
    callback: CallbackQuery,
    bot: Bot,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Resume the interrupted import graph with the user's yes/no decision."""
    if not _is_allowed(callback, settings):
        await callback.answer("⛔ Доступ запрещён.", show_alert=True)
        return
    await callback.answer()  # dismiss the loading spinner

    message = callback.message
    if callback.data is None or not isinstance(message, Message):
        return

    confirmed = callback.data == IMPORT_CONFIRM_YES
    chat_id = message.chat.id
    user_id = callback.from_user.id if callback.from_user else 0
    thread_id = f"tg:{chat_id}"

    # Замораживаем карточку, чтобы по ней нельзя было кликнуть дважды.
    decision_text = "✅ Импортирую…" if confirmed else "🛑 Импорт отменён."
    with contextlib.suppress(TelegramBadRequest):
        await message.edit_text(decision_text, reply_markup=None)

    callbacks = cast("list[BaseCallbackHandler]", [make_callback_handler()])
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": {
            "telegram_user_id": str(user_id),
            "telegram_chat_id": str(chat_id),
            "langfuse_user_id": str(user_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["telegram", "phase1", "import_confirm"],
            "langfuse_trace_name": "telegram-import-confirm",
        },
    }

    try:
        result = await graph.ainvoke(Command(resume=confirmed), config=config)
    except Exception:
        logger.exception("import confirm resume failed")
        await bot.send_message(chat_id, "⚠️ Не смог завершить импорт. Подробности в логах.")
        return

    # На отмене замороженная карточка («🛑 Импорт отменён.») уже и есть
    # подтверждение — текст графа продублировал бы его второй репликой (QA-13).
    # Дальнейших действий у отмены нет: ни вставок, ни уточнений.
    if confirmed:
        await send_import_result(message, result)
