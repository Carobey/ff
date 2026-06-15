"""
LangFuse bootstrap. Создаёт CallbackHandler для LangChain/LangGraph.

Под лекцию 28 мая — observability и evals. Через callback автоматически попадают:
- LLM calls (input/output/tokens/latency/cost)
- LangGraph nodes (transitions, state diff)
- Tool calls
- Errors

Доп. — кастомные spans через @observe декоратор для произвольной логики.
"""

from __future__ import annotations

import logging
from typing import Literal

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from family_finance.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    """Singleton клиент LangFuse."""
    global _langfuse_client
    if _langfuse_client is None:
        s = get_settings()
        _langfuse_client = Langfuse(
            public_key=s.langfuse_public_key.get_secret_value(),
            secret_key=s.langfuse_secret_key.get_secret_value(),
            host=s.langfuse_host,
        )
        logger.info("✅ LangFuse client initialized: %s", s.langfuse_host)
    return _langfuse_client


def make_callback_handler() -> CallbackHandler:
    """
    Создать CallbackHandler для одного LangGraph invoke.

    LangFuse v4 читает user/session/tags из config["metadata"]:
        langfuse_user_id, langfuse_session_id, langfuse_tags, langfuse_trace_name
    """
    s = get_settings()
    get_langfuse()
    return CallbackHandler(
        public_key=s.langfuse_public_key.get_secret_value(),
    )


def emit_score(
    name: str,
    value: float,
    *,
    comment: str | None = None,
    data_type: Literal["NUMERIC", "BOOLEAN"] = "NUMERIC",
) -> None:
    """Прикрепить production-скор к ТЕКУЩЕМУ trace (внутри graph.ainvoke).

    Используется нодами графа для бизнес-метрик дашбордов:
    injection_blocked, categorization_review_rate и т.п.

    Observability никогда не должна ронять приложение — любая ошибка
    (нет активного trace, сеть, версия SDK) глотается с warning-логом.
    """
    try:
        get_langfuse().score_current_trace(
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
        )
    except Exception:
        logger.warning("langfuse_emit_score_failed name=%s", name, exc_info=True)


def flush() -> None:
    """Гарантированно отправить все pending traces. Вызывать в shutdown handler."""
    if _langfuse_client is not None:
        _langfuse_client.flush()
