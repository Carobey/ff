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


def flush() -> None:
    """Гарантированно отправить все pending traces. Вызывать в shutdown handler."""
    if _langfuse_client is not None:
        _langfuse_client.flush()
