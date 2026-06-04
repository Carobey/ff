"""APScheduler integration: per-member weekly-digest jobs.

Each family member with a non-null ``digest_cron`` gets their own cron
job (id = ``digest:<member_id>``). When the user changes their schedule
via ``/digest_schedule``, the handler calls :func:`schedule_for_member`,
which removes the existing job (if any) and adds a new one with the new
cron expression. ``/digest_off`` calls :func:`unschedule_member`.

The scheduler instance is stashed on the aiogram Dispatcher's
``workflow_data`` so command handlers can mutate it.
"""

from __future__ import annotations

import uuid

import structlog
from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from family_finance.agents.digest import build_digest
from family_finance.infrastructure.persistence import PostgresTransactionRepository

logger = structlog.get_logger()


def _job_id(member_id: uuid.UUID) -> str:
    return f"digest:{member_id}"


async def _send_digest(bot: Bot, family_id_str: str, chat_id: int) -> None:
    """Job body — build the digest for one family and send it to one chat."""
    family_id = uuid.UUID(family_id_str)
    try:
        text = await build_digest(family_id)
    except Exception:
        logger.exception("digest.build_failed", family_id=family_id_str)
        return
    if text is None:
        logger.info("digest.skip_empty", family_id=family_id_str)
        return
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        logger.exception("digest.send_failed", family_id=family_id_str, chat_id=chat_id)


def schedule_for_member(
    scheduler: AsyncIOScheduler,
    bot: Bot,
    *,
    family_id: uuid.UUID,
    member_id: uuid.UUID,
    telegram_user_id: int,
    cron: str,
) -> None:
    """Register or replace one member's digest job with the given cron string.

    *cron* is the 5-field representation from
    :meth:`DigestSchedule.to_cron` — ``"M H * * dow"``.
    """
    minute, hour, dom, mon, dow = cron.split()
    trigger = CronTrigger(
        minute=minute,
        hour=hour,
        day=dom,
        month=mon,
        day_of_week=dow,
        timezone="Europe/Moscow",
    )
    scheduler.add_job(
        _send_digest,
        trigger=trigger,
        kwargs={
            "bot": bot,
            "family_id_str": str(family_id),
            "chat_id": telegram_user_id,
        },
        id=_job_id(member_id),
        replace_existing=True,
    )
    logger.info(
        "scheduler.member_scheduled",
        member_id=str(member_id),
        cron=cron,
    )


def unschedule_member(scheduler: AsyncIOScheduler, member_id: uuid.UUID) -> bool:
    """Remove a member's job. Returns True if a job was actually removed."""
    job_id = _job_id(member_id)
    if scheduler.get_job(job_id) is None:
        return False
    scheduler.remove_job(job_id)
    logger.info("scheduler.member_unscheduled", member_id=str(member_id))
    return True


async def start_scheduler(bot: Bot) -> AsyncIOScheduler:
    """Start AsyncIO scheduler and replay every persisted schedule from DB.

    Caller owns the lifecycle — shutdown should call ``scheduler.shutdown()``.
    """
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.start()

    repo = PostgresTransactionRepository()
    try:
        rows = await repo.iter_digest_schedules()
    except Exception:
        logger.exception("scheduler.replay_failed")
        rows = []

    for family_id, member_id, telegram_user_id, cron in rows:
        try:
            schedule_for_member(
                scheduler,
                bot,
                family_id=family_id,
                member_id=member_id,
                telegram_user_id=telegram_user_id,
                cron=cron,
            )
        except Exception:
            logger.exception(
                "scheduler.replay_member_failed",
                member_id=str(member_id),
                cron=cron,
            )

    logger.info("scheduler.started", restored=len(rows))
    return scheduler
