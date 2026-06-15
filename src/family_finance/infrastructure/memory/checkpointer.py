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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from family_finance.domain import (
    Category,
    Currency,
    Direction,
    Transaction,
    TransactionSource,
)
from family_finance.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)

# Доменные типы, которые попадают в state (`parsed_transactions`) и, значит,
# сериализуются в чекпойнт. LangGraph 1.2+ требует явно разрешать кастомные
# типы при десериализации (иначе — deprecation warning, в будущем — блокировка).
# Stdlib-типы (Decimal, datetime, UUID) уже в SAFE_MSGPACK_TYPES — их не перечисляем.
_ALLOWED_CHECKPOINT_TYPES = (
    Transaction,
    Category,
    Currency,
    Direction,
    TransactionSource,
)


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

    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES)
    async with AsyncPostgresSaver.from_conn_string(conn_string, serde=serde) as saver:
        await saver.setup()  # создаёт таблицы checkpointer'а если их нет
        logger.info(
            "✅ PostgresSaver ready (msgpack allowlist: %d custom types: %s)",
            len(_ALLOWED_CHECKPOINT_TYPES),
            ", ".join(t.__name__ for t in _ALLOWED_CHECKPOINT_TYPES),
        )
        yield saver
