"""Inline-button handler for clarification questions.

Когда ingest+categorizer возвращает open_questions,
documents.py отправляет сообщения с inline-кнопками.
Клик по кнопке → callback_query → synthezises "N category" text → обычный clarify flow.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import Settings

logger = structlog.get_logger()
router = Router(name="clarify_buttons")

# Callback data format: "clarify:{question_id}:{category_value}"
_PREFIX = "clarify"


def _is_allowed(callback: CallbackQuery, settings: Settings) -> bool:
    allowed = settings.allowed_user_ids
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER_IDS пуст — доступ закрыт")
        return False
    user_id = callback.from_user.id if callback.from_user else 0
    return user_id in allowed


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
    if not _is_allowed(callback, settings):
        await callback.answer("⛔ Доступ запрещён.", show_alert=True)
        return
    await callback.answer()  # dismiss the loading spinner

    message = callback.message
    if callback.data is None or not isinstance(message, Message):
        return

    parsed = parse_callback_data(callback.data)
    if parsed is None:
        return

    question_id, category_value = parsed
    # Synthesize the text the user would have typed, e.g. "1 одежда"
    synthetic_text = f"{question_id} {category_value}"

    user_id = callback.from_user.id if callback.from_user else 0
    display_name = callback.from_user.full_name if callback.from_user else str(user_id)
    chat_id = message.chat.id
    thread_id = f"tg:{chat_id}"

    repository = PostgresTransactionRepository()
    family_id, member_id = await repository.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )

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
            {
                "messages": [HumanMessage(content=synthetic_text)],
                "telegram_user_id": user_id,
                "telegram_chat_id": chat_id,
                "family_id": str(family_id),
                "member_id": str(member_id),
            },
            config=config,
        )
    except Exception:
        logger.exception("clarify button failed")
        await bot.send_message(chat_id, "⚠️ Не смог применить уточнение.")
        return

    reply = str(result["messages"][-1].content)

    # Перерисовываем сам вопрос на месте: ✅ + выбранная категория, без клавиатуры.
    # Так видно, на что уже ответили, и нет кучи отдельных «Принял уточнения».
    question_text = _question_text(message.text or "")
    if category_value == "__lookup__":
        # Веб-поиск: показываем что нашли (строку-счётчик «Принял уточнения» убираем).
        detail = "\n".join(
            ln for ln in reply.splitlines() if not ln.startswith("Принял уточнения")
        ).strip()
        confirm = f"✅ {question_text}\n{detail}" if detail else f"✅ {question_text}"
    else:
        label = _chosen_label(message, callback.data) or category_value
        confirm = f"✅ {question_text} → {label}"

    try:
        await message.edit_text(confirm, reply_markup=None)
    except TelegramBadRequest:
        # Сообщение слишком старое для редактирования → обычный ответ.
        await bot.send_message(chat_id, confirm)


def _question_text(message_text: str) -> str:
    """Текст вопроса без строки-подсказки про ручной ввод (она начинается с 💬)."""
    return "\n".join(ln for ln in message_text.splitlines() if not ln.startswith("💬")).strip()


def _chosen_label(message: Message, data: str) -> str | None:
    """Достать подпись нажатой кнопки из живой клавиатуры (без дублей карт меток)."""
    markup = message.reply_markup
    if markup is None:
        return None
    for row in markup.inline_keyboard:
        for button in row:
            if button.callback_data == data:
                return button.text
    return None
