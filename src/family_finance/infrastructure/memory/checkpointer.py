"""
LangGraph PostgresSaver — working memory.

PostgresSaver хранит state каждого узла графа после каждого шага.
Это:
- Resume agent после рестарта (важно для long-running)
- Human-in-the-loop через interrupt()
- Debug — точка возврата к любому шагу

Используем AsyncPostgresSaver (LangGraph 0.4+).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from family_finance.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """
    Context manager — поднимает PostgresSaver, гарантирует cleanup.

    Использовать в bot startup:
        async with get_checkpointer() as saver:
            graph = build_supervisor(saver)
            ...
    """
    s = get_settings()
    # PostgresSaver требует psycopg-стиль URL (не asyncpg)
    conn_string = s.database_url.get_secret_value()
    logger.info("Connecting checkpointer: %s", conn_string.split("@")[-1])

    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        await saver.setup()  # создаёт таблицы checkpointer'а если их нет
        logger.info("✅ PostgresSaver ready")
        yield saver
