"""
Start / message handlers.

Phase 0:
- /start приветствует
- любое текстовое сообщение → LangGraph supervisor → ответ

В каждый invoke прокидывается LangFuse callback с user_id+session_id+tags —
это то что хочет увидеть Эмиль на ревью 4 июня.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from family_finance.agents.advisor import goal_status_text
from family_finance.agents.budgets import (
    current_moscow_month,
    format_budgets,
    parse_budget_category,
)
from family_finance.agents.digest import build_digest
from family_finance.agents.digest_schedule_parser import parse_digest_schedule
from family_finance.bot.scheduler import schedule_for_member, unschedule_member
from family_finance.bot.telegram_text import answer_plain
from family_finance.infrastructure.observability import make_callback_handler
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import Settings

logger = logging.getLogger(__name__)
router = Router(name="start")


def _is_allowed(message: Message, settings: Settings) -> bool:
    """Whitelist по telegram_user_id. Без этого — любой может писать боту."""
    allowed = settings.allowed_user_ids
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USER_IDS пуст — доступ закрыт")
        return False
    user_id = message.from_user.id if message.from_user else 0
    return user_id in allowed


@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён. Обратись к администратору.")
        return
    await message.answer(
        "👋 Привет! Я — финансовый помощник семьи.\n\n"
        "Что умею:\n"
        "• <b>CSV/PDF выписки</b> — пришли файлом, я разберу и категоризирую\n"
        "• <b>Фото чека</b> — найду QR, заберу детализацию из ФНС\n"
        "• <b>Вопросы по тратам</b> — «сколько на еду в мае?»\n"
        "• <b>Паттерны</b> — «когда я последний раз так часто заказывал доставку?»\n"
        "• <b>/subscriptions</b> — список регулярных трат\n"
        "• <b>/budget продукты 30000</b> — установить месячный бюджет\n"
        "• <b>/budgets</b> — статус всех бюджетов (зелёный/жёлтый/красный)\n"
        "• <b>/budget_off продукты</b> — убрать бюджет\n"
        "• <b>Совет наставника</b> — «на чём сэкономить?», «как копить?»\n"
        "• <b>/goal 200000 до 31.12.2026</b> — цель накопления + прогресс\n"
        "• <b>/goal_off</b> — убрать цель\n"
        "• <b>/digest</b> — недельная сводка прямо сейчас\n"
        "• <b>/digest_schedule по воскресеньям в 19:00</b> — расписание авто-пуша\n"
        "• <b>/digest_off</b> — отключить авто-пуш\n"
    )


async def _invoke_graph(
    message: Message,
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    user_text: str,
    trace_name: str,
) -> None:
    """Common path: feed *user_text* into the supervisor graph and reply."""
    user_id = message.from_user.id if message.from_user else 0
    thread_id = f"tg:{message.chat.id}"  # один thread на чат = диалог сохраняется
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    family_id, member_id = await PostgresTransactionRepository().ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )

    callbacks = cast("list[BaseCallbackHandler]", [make_callback_handler()])
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": {
            "telegram_user_id": str(user_id),
            "telegram_chat_id": str(message.chat.id),
            "langfuse_user_id": str(user_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["telegram"],
            "langfuse_trace_name": trace_name,
        },
    }

    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "telegram_user_id": user_id,
                "telegram_chat_id": message.chat.id,
                "family_id": str(family_id),
                "member_id": str(member_id),
            },
            config=config,
        )
    except Exception:
        logger.exception("graph.invoke failed")
        await message.answer("⚠️ Не смог обработать сообщение. Подробности уже в логах.")
        return

    reply = result["messages"][-1].content
    await answer_plain(message, reply)


@router.message(Command("subscriptions"))
async def cmd_subscriptions(
    message: Message,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Show detected recurring expenses for the family."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return
    await _invoke_graph(
        message,
        graph,
        user_text="мои подписки",
        trace_name="telegram-subscriptions",
    )


@router.message(Command("digest"))
async def cmd_digest(message: Message, settings: Settings) -> None:
    """Send the weekly digest on demand."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    family_id, _ = await PostgresTransactionRepository().ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    text = await build_digest(family_id)
    if text is None:
        await message.answer("За последнюю неделю расходов не было — нечего показать.")
        return
    await message.answer(text)


@router.message(Command("digest_schedule"))
async def cmd_digest_schedule(
    message: Message,
    command: CommandObject,
    settings: Settings,
    scheduler: AsyncIOScheduler,
    bot: Bot,
) -> None:
    """Set when the weekly digest should arrive. Free-form Russian/English.

    Examples::

        /digest_schedule по воскресеньям в 19:00
        /digest_schedule пятница 20:30
        /digest_schedule sun 10am
    """
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Скажи, когда присылать дайджест. Например:\n"
            "<code>/digest_schedule по воскресеньям в 19:00</code>\n"
            "<code>/digest_schedule пятница 20:30</code>"
        )
        return

    schedule = await parse_digest_schedule(args)
    if schedule is None:
        await message.answer(
            "Не понял время. Скажи один день недели и время, например «по воскресеньям в 19:00»."
        )
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    cron = schedule.to_cron()
    await repo.set_digest_cron(member_id=member_id, cron=cron)
    schedule_for_member(
        scheduler,
        bot,
        family_id=family_id,
        member_id=member_id,
        telegram_user_id=user_id,
        cron=cron,
    )
    await message.answer(f"✅ Буду присылать дайджест {schedule.human_label()} (МСК).")


@router.message(Command("budget"))
async def cmd_budget(message: Message, command: CommandObject, settings: Settings) -> None:
    """Set a monthly budget for one category.

    Usage::

        /budget продукты 30000
        /budget food.groceries 30000
    """
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Укажи категорию и сумму. Например:\n"
            "<code>/budget продукты 30000</code>\n"
            "<code>/budget food.groceries 30000</code>"
        )
        return

    parts = args.rsplit(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Нужны категория и сумма, например: <code>/budget еда 30000</code>")
        return
    raw_category, raw_amount = parts
    category = parse_budget_category(raw_category)
    if category is None:
        await message.answer(
            f"Не понял категорию «{raw_category}». Скажи русским словом "
            "(продукты, еда, ЖКХ, аптека…) или точным enum (food.groceries)."
        )
        return
    try:
        amount = Decimal(raw_amount.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        await message.answer(f"Не понял сумму «{raw_amount}» — должна быть числом.")
        return
    if amount <= 0:
        await message.answer("Сумма должна быть положительной.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    await repo.set_budget(family_id=family_id, category=category, monthly_limit=amount)
    await message.answer(
        f"✅ Бюджет на «{category.value}» установлен: {int(amount):,} ₽ / месяц".replace(",", " ")
    )


@router.message(Command("budgets"))
async def cmd_budgets(message: Message, settings: Settings) -> None:
    """Show all configured budgets and current month's spend."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    month_start, month_end = current_moscow_month()
    statuses = await repo.get_budget_status(
        family_id=family_id,
        month_start=month_start,
        month_end=month_end,
    )
    await message.answer(format_budgets(statuses))


@router.message(Command("budget_off"))
async def cmd_budget_off(message: Message, command: CommandObject, settings: Settings) -> None:
    """Remove a budget for one category."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    args = (command.args or "").strip()
    if not args:
        await message.answer("Скажи какую категорию убрать, например: <code>/budget_off еда</code>")
        return
    category = parse_budget_category(args)
    if category is None:
        await message.answer(f"Не понял категорию «{args}».")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    removed = await repo.delete_budget(family_id=family_id, category=category)
    msg = f"🛑 Бюджет на «{category.value}» снят." if removed else "Такого бюджета и не было."
    await message.answer(msg)


def _parse_goal_date(raw: str) -> date | None:
    """Parse a target date in ``YYYY-MM-DD`` or ``DD.MM.YYYY`` form."""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


@router.message(Command("goal"))
async def cmd_goal(message: Message, command: CommandObject, settings: Settings) -> None:
    """Set or show the family's savings goal.

    Usage::

        /goal                       — показать текущую цель и прогресс
        /goal 200000                — цель без срока
        /goal 200000 до 31.12.2026  — цель к дате
    """
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )

    args = (command.args or "").strip()
    if not args:
        await message.answer(await goal_status_text(family_id))
        return

    amount_part, _, date_part = args.partition(" до ")
    target_date: date | None = None
    if date_part.strip():
        target_date = _parse_goal_date(date_part.strip())
        if target_date is None:
            await message.answer("Не понял дату. Формат: <code>/goal 200000 до 31.12.2026</code>")
            return
    try:
        amount = Decimal(amount_part.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        await message.answer(f"Не понял сумму «{amount_part.strip()}» — должна быть числом.")
        return
    if amount <= 0:
        await message.answer("Сумма должна быть положительной.")
        return

    await repo.set_savings_goal(
        family_id=family_id,
        target_amount=amount,
        target_date=target_date,
    )
    await message.answer("✅ Цель сохранена.\n\n" + await goal_status_text(family_id))


@router.message(Command("goal_off"))
async def cmd_goal_off(message: Message, settings: Settings) -> None:
    """Remove the family's savings goal."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    family_id, _ = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    removed = await repo.delete_savings_goal(family_id=family_id)
    await message.answer("🛑 Цель снята." if removed else "Цели и так не было.")


@router.message(Command("digest_off"))
async def cmd_digest_off(
    message: Message,
    settings: Settings,
    scheduler: AsyncIOScheduler,
) -> None:
    """Cancel the scheduled digest for the calling member."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    user_id = message.from_user.id if message.from_user else 0
    display_name = message.from_user.full_name if message.from_user else str(user_id)
    repo = PostgresTransactionRepository()
    _, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=user_id,
        name=display_name,
    )
    await repo.set_digest_cron(member_id=member_id, cron=None)
    removed = unschedule_member(scheduler, member_id)
    msg = "🛑 Дайджест отключён." if removed else "Дайджест и так не был включён."
    await message.answer(msg)


@router.message(F.text)
async def handle_any_message(
    message: Message,
    settings: Settings,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Любое текстовое сообщение → supervisor."""
    if not _is_allowed(message, settings):
        await message.answer("⛔ Доступ запрещён.")
        return

    if not message.text:
        await message.answer("Пока понимаю только текст / файлы / фото.")
        return

    await _invoke_graph(message, graph, user_text=message.text, trace_name="telegram-message")
