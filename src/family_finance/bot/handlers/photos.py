"""Photo handler: receipt QR scan (P2-09)."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from aiogram import Bot, F, Router
from aiogram.types import Message
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.bot.telegram_text import answer_plain
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import Settings

logger = logging.getLogger(__name__)
router = Router(name="photos")


def _is_allowed(message: Message, settings: Settings) -> bool:
    allowed = settings.allowed_user_ids
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER_IDS пуст — доступ закрыт")
        return False
    user_id = message.from_user.id if message.from_user else 0
    return user_id in allowed


@router.message(F.photo)
async def handle_photo(
    message: Message,
    bot: Bot,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Receive a photo, save the highest-resolution version, pass to receipt_node."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    if not message.photo:
        await message.answer("Фото не найдено в сообщении.")
        return

    # Telegram sends several sizes — take the largest (last in list)
    photo = message.photo[-1]
    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)

    repository = PostgresTransactionRepository()
    family_id, member_id = await repository.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )

    upload_dir = Path("uploads") / str(message.chat.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{photo.file_unique_id}.jpg"

    tg_file = await bot.get_file(photo.file_id)
    if tg_file.file_path is None:
        await message.answer("Telegram не вернул путь к файлу. Попробуй ещё раз.")
        return

    buffer = BytesIO()
    await bot.download_file(tg_file.file_path, buffer)
    saved_path.write_bytes(buffer.getvalue())

    await message.answer("📷 Фото получено, ищу QR-код…")

    thread_id = f"tg:{message.chat.id}"
    callbacks = cast("list[BaseCallbackHandler]", [make_callback_handler()])
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": {
            "telegram_user_id": str(user_id),
            "telegram_chat_id": str(message.chat.id),
            "langfuse_user_id": str(user_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["telegram", "phase2", "receipt"],
            "langfuse_trace_name": "telegram-receipt-scan",
        },
    }

    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Фото чека для сканирования QR")],
                "telegram_user_id": user_id,
                "telegram_chat_id": message.chat.id,
                "family_id": str(family_id),
                "member_id": str(member_id),
                "pending_photo": str(saved_path),
            },
            config=config,
        )
    except Exception:
        logger.exception("receipt photo handling failed")
        await message.answer("⚠️ Не смог обработать фото. Подробности в логах.")
        return

    reply = result["messages"][-1].content
    await answer_plain(message, reply)
