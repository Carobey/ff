"""
Telegram bot entry. aiogram 3.x.

Стартует:
- Bot + Dispatcher
- LangGraph supervisor с PostgresSaver
- LangFuse callback handler

При shutdown — flush LangFuse + закрывает БД.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
from contextlib import suppress

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiohttp import AsyncResolver, ClientSession

from family_finance.bot.handlers import register_all_handlers
from family_finance.bot.scheduler import start_scheduler
from family_finance.infrastructure.mcp.client import close_finance_tools, get_finance_tools
from family_finance.infrastructure.memory import get_checkpointer
from family_finance.infrastructure.memory.graphiti_client import graphiti_init
from family_finance.infrastructure.observability import flush, get_langfuse
from family_finance.infrastructure.settings import get_settings


def setup_logging(level: str = "INFO") -> None:
    """structlog → stdlib logging."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


async def amain() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    log = structlog.get_logger()

    log.info("startup.begin", service=settings.service_name, env=settings.environment)

    # Прогреваем LangFuse (проверяем что отвечает)
    get_langfuse()

    # Force IPv4 + use public DNS for the Telegram session (dev-only, gated).
    #
    # Why: on some dev networks the local resolver (router / Yandex DNS) returns
    # ONLY AAAA records for api.telegram.org, and the resulting IPv6 address
    # is not routable. ``family=AF_INET`` alone isn't enough — getaddrinfo
    # then returns nothing and raises EAI_AGAIN ("Temporary failure in name
    # resolution"). The AsyncResolver bypasses the local resolver entirely
    # and queries Cloudflare/Google directly. Off by default — opt in via
    # ``TELEGRAM_FORCE_IPV4_DNS=true`` only where the network needs it.
    class _TelegramSession(AiohttpSession):
        async def create_session(self) -> ClientSession:
            resolver = AsyncResolver(nameservers=["1.1.1.1", "8.8.8.8"])
            self._connector_init = {
                **(self._connector_init or {}),
                "family": socket.AF_INET,
                "resolver": resolver,
            }
            return await super().create_session()

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=_TelegramSession() if settings.telegram_force_ipv4_dns else None,
    )
    dp = Dispatcher()

    # Graphiti indices (FalkorDB) — idempotent, fast if already exist
    try:
        await graphiti_init()
    except Exception:
        log.warning("graphiti.init_failed", hint="docker compose --profile phase2 up -d falkordb")

    # PostgresSaver — context manager. Граф живёт пока поднят бот.
    async with get_checkpointer() as checkpointer:
        # Импорт здесь — чтобы checkpointer уже был готов
        from family_finance.agents import build_supervisor_graph

        graph = build_supervisor_graph(checkpointer)

        # Прогреваем MCP-сессию в главной задаче: stdio-сессия живёт один subprocess
        # и не выдерживает конкурентного открытия из параллельных воркеров (веер
        # Send, ADR 0008). Открыв её здесь, мы (а) убираем гонку ленивой инициализации
        # и (б) открываем и закрываем cancel scope в одной задаче — чистый shutdown.
        try:
            await get_finance_tools()
        except Exception:
            log.warning("mcp.warmup_failed", hint="проверь family_finance.mcp_server.server")

        # Передаём graph через workflow_data — доступен в хендлерах через data["graph"]
        dp["graph"] = graph
        dp["settings"] = settings

        register_all_handlers(dp)

        log.info("startup.ready", bot=(await bot.get_me()).username)

        scheduler = await start_scheduler(bot)
        dp["scheduler"] = scheduler

        try:
            await dp.start_polling(bot)
        finally:
            log.info("shutdown.scheduler")
            scheduler.shutdown(wait=False)
            log.info("shutdown.mcp")
            await close_finance_tools()
            log.info("shutdown.flush_langfuse")
            flush()
            log.info("shutdown.done")


def main() -> None:
    with suppress(KeyboardInterrupt, SystemExit):
        asyncio.run(amain())


if __name__ == "__main__":
    main()
