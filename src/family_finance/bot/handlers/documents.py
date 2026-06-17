"""Document handlers for bank statement uploads."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import structlog
from aiogram import Bot, F, Router
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.agents.clarifications import ClarificationQuestion
from family_finance.bot.handlers.clarify_buttons import make_callback_data
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import Settings

logger = structlog.get_logger()
router = Router(name="documents")


def _is_allowed(message: Message, settings: Settings) -> bool:
    allowed = settings.allowed_user_ids
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER_IDS пуст — доступ закрыт")
        return False
    user_id = message.from_user.id if message.from_user else 0
    return user_id in allowed


@router.message(F.document)
async def handle_document(
    message: Message,
    bot: Bot,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Accept Tinkoff CSV or Sberbank PDF statements and pass them to the ingest node."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    document = message.document
    if document is None:
        await message.answer("Файл не найден в сообщении.")
        return

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if document.file_size is not None and document.file_size > max_bytes:
        await message.answer(f"⛔ Файл больше {settings.max_upload_mb} МБ — не принимаю.")
        return

    filename = document.file_name or document.file_unique_id
    fname_lower = filename.lower()
    is_csv_by_ext = fname_lower.endswith(".csv")
    is_pdf_by_ext = fname_lower.endswith(".pdf")
    if not (is_csv_by_ext or is_pdf_by_ext):
        await message.answer("Принимаю:\n• CSV — выписка Тинькофф\n• PDF — выписка Сбербанк")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repository = PostgresTransactionRepository()
    family_id, member_id = await repository.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )

    # Download first so we can sniff the content
    upload_dir = Path("uploads") / str(message.chat.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    tg_file = await bot.get_file(document.file_id)
    if tg_file.file_path is None:
        await message.answer("Telegram не вернул путь к файлу. Попробуй отправить ещё раз.")
        return
    buffer = BytesIO()
    await bot.download_file(tg_file.file_path, buffer)
    content_bytes = buffer.getvalue()

    # Content-sniff: real PDFs start with %PDF-; anything else is treated as CSV.
    # Strip a leading UTF-8 BOM first — some tools prepend it even to PDFs.
    sniff = content_bytes.lstrip(b"\xef\xbb\xbf")[:5]
    is_pdf = sniff == b"%PDF-"
    is_csv = not is_pdf

    ext = ".pdf" if is_pdf else ".csv"
    saved_path = upload_dir / f"{document.file_unique_id}{ext}"
    saved_path.write_bytes(content_bytes)

    bank_label = "Тинькофф CSV" if is_csv else "Сбербанк PDF"
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
            "langfuse_tags": ["telegram", "phase1", "csv" if is_csv else "pdf"],
            "langfuse_trace_name": "telegram-statement-import",
        },
    }

    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=f"Загружена выписка ({bank_label}): {filename}")],
                "telegram_user_id": user_id,
                "telegram_chat_id": message.chat.id,
                "family_id": str(family_id),
                "member_id": str(member_id),
                # Always reset both keys so stale checkpoint values don't bleed through
                "pending_csv": str(saved_path) if is_csv else None,
                "pending_pdf": str(saved_path) if is_pdf else None,
            },
            config=config,
        )
    except Exception:
        logger.exception("statement import failed")
        await message.answer("⚠️ Не смог импортировать выписку. Подробности уже в логах.")
        return

    await send_import_result(message, result)


# ── Import result surfacing (interrupt card / reply + questions) ───────────────

# Callback data for the HITL import-confirmation buttons (см. import_confirm_buttons.py).
IMPORT_CONFIRM_YES = "import:yes"
IMPORT_CONFIRM_NO = "import:no"


def import_confirm_keyboard() -> InlineKeyboardMarkup:
    """Two-button keyboard: confirm or cancel a pending bulk import."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Импортировать", callback_data=IMPORT_CONFIRM_YES),
                InlineKeyboardButton(text="🛑 Отмена", callback_data=IMPORT_CONFIRM_NO),
            ]
        ]
    )


def format_import_preview(payload: dict[str, Any]) -> str:
    """Render the ``interrupt()`` payload from ingest into a confirmation card."""
    return (
        f"📄 Разобрал выписку ({payload.get('bank')}).\n"
        f"Нашёл <b>{payload.get('count')}</b> операций "
        f"за период {payload.get('period_start')} — {payload.get('period_end')}.\n\n"
        "Импортировать в базу?"
    )


async def send_import_result(message: Message, result: dict[str, Any]) -> None:
    """Surface a graph result that may be paused on the import-confirm interrupt.

    Paused (``__interrupt__`` present) → show the preview card with confirm buttons.
    Otherwise → normal reply text plus inline-keyboard clarification questions.
    """
    interrupts = result.get("__interrupt__")
    if interrupts:
        await message.answer(
            format_import_preview(interrupts[0].value),
            reply_markup=import_confirm_keyboard(),
        )
        return

    reply = result["messages"][-1].content
    await message.answer(str(reply))

    # Send inline-keyboard questions for needs_review transactions
    open_questions: list[ClarificationQuestion] = result.get("open_questions") or []
    for question in open_questions:
        if not question.get("import_hashes"):
            continue  # summary placeholder
        qid = question["id"]
        hint = (
            f"💬 Не та категория на кнопках? Напиши своими словами: «{qid} спортзал» "
            f"или «{qid} не знаю» — поищу в интернете."
        )
        await message.answer(
            f"{question['text']}\n\n{hint}",
            reply_markup=_build_question_keyboard(question),
        )


# ── Category buttons ──────────────────────────────────────────────────────────

_COMMON_BUTTONS: list[tuple[str, str]] = [
    ("🛒 Продукты", "food.groceries"),
    ("🍔 Кафе", "food.restaurant"),
    ("🚕 Такси", "transport.taxi"),
    ("⛽ АЗС", "transport.fuel"),
    ("🚇 Транспорт", "transport.public"),
    ("🛍 Покупки", "shopping.generic"),
    ("👕 Одежда", "shopping.clothes"),
    ("🏠 ЖКХ", "home.utilities"),
    ("📱 Связь", "home.telecom"),
    ("💊 Аптека", "health.pharmacy"),
    ("🏥 Медицина", "health.generic"),
    ("🧒 Дети", "kids.clothes"),
    ("🎬 Развлечения", "entertainment.events"),
    ("📺 Подписки", "entertainment.subscriptions"),
    ("💄 Красота", "beauty.care"),
    ("↔️ Перевод", "transfer.internal"),
    ("🔎 Не знаю", "__lookup__"),
    ("❓ Непонятно", "unclassified"),
]

_TRANSFER_BUTTONS: list[tuple[str, str]] = [
    ("↔️ Перевод между своими", "transfer.internal"),
    ("💳 Расход", "shopping.generic"),
    ("💰 Доход", "income.other"),
]


def _build_question_keyboard(question: ClarificationQuestion) -> InlineKeyboardMarkup:
    """Build inline keyboard for one clarification question."""
    qid = question["id"]
    buttons = _TRANSFER_BUTTONS if question.get("reason") == "перевод" else _COMMON_BUTTONS
    rows: list[list[InlineKeyboardButton]] = []
    # 3 buttons per row
    row: list[InlineKeyboardButton] = []
    for label, cat_value in buttons:
        row.append(
            InlineKeyboardButton(
                text=label,
                callback_data=make_callback_data(qid, cat_value),
            )
        )
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)
