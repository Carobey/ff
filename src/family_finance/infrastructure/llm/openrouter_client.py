"""
LLM-адаптер. Используем OpenRouter через first-party `langchain-openrouter` —
единый API ко всем провайдерам (OpenAI, Anthropic, Google, Meta, DeepSeek, ...).

Почему НЕ ChatOpenAI с base_url=openrouter.ai/api/v1:
LangChain официально DEPRECATED этот путь в 2026 (см. ADR 0004).
ChatOpenRouter использует официальный OpenRouter SDK — корректно работают:
- Structured output (tool calling, response_format)
- Provider-specific fields (reasoning_content для o-моделей и Claude thinking)
- Fallback routing (OpenRouter попробует следующую модель при ошибке)

Стратегия моделей (май 2026):
  supervisor — openai/gpt-5.4              ($2.50/$15.00) routing+reasoning
  worker     — google/gemini-2.5-flash     ($0.075/$0.30) категоризация/parsing

Cascade pattern: дешёвый worker делает работу, supervisor только для неоднозначностей.
Это main lever экономии.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openrouter import ChatOpenRouter

from family_finance.infrastructure.security import mask_messages
from family_finance.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)


class _MaskingRunnable[T]:
    """Wrap a structured-output runnable so prompt PII is masked before invoke."""

    def __init__(self, inner: Runnable[LanguageModelInput, T]) -> None:
        self._inner = inner

    async def ainvoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        return await self._inner.ainvoke(mask_messages(input), config, **kwargs)


class MaskingChatModel:
    """Chat-model facade that masks PII on every outbound call.

    This is the single LLM chokepoint (CLAUDE.md «Адаптер ОДИН»): wrapping it
    means each node's prompt is anonymized before it reaches OpenRouter without
    any per-node change. Only the methods the nodes use are exposed —
    ``ainvoke`` and ``with_structured_output``.
    """

    def __init__(self, inner: ChatOpenRouter) -> None:
        self._inner = inner

    async def ainvoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        return await self._inner.ainvoke(mask_messages(input), config, **kwargs)

    def with_structured_output[T](self, schema: type[T], **kwargs: Any) -> _MaskingRunnable[T]:
        inner = cast(
            "Runnable[LanguageModelInput, T]",
            self._inner.with_structured_output(cast("Any", schema), **kwargs),
        )
        return _MaskingRunnable(inner)


@lru_cache(maxsize=4)
def get_chat_model(*, tier: str = "worker", temperature: float = 0.1) -> MaskingChatModel:
    """
    Получить chat-модель через OpenRouter. Кешируется по tier+temperature.

    Возвращает обёртку :class:`MaskingChatModel` — она маскирует PII в тексте
    пользователя перед отправкой в облачный LLM (см. infrastructure/security).

    tier:
        "supervisor" — primary: gpt-5.4 (с fallbacks)
        "worker"     — primary: gemini-2.5-flash (с fallbacks)
    """
    s = get_settings()

    if tier == "supervisor":
        model_name = s.llm_supervisor_model
        fallbacks = s.supervisor_fallback_list
    else:
        model_name = s.llm_worker_model
        fallbacks = s.worker_fallback_list

    logger.info(
        "Creating ChatOpenRouter: tier=%s model=%s fallbacks=%s",
        tier,
        model_name,
        fallbacks,
    )

    # OpenRouter-specific params accepted by the installed SDK.
    # The SDK has no `extra_body` passthrough, so keep these as typed fields.
    model_kwargs: dict[str, object] = {}
    if fallbacks:
        # OpenRouter попробует следующую если основная недоступна
        # https://openrouter.ai/docs/features/model-routing
        model_kwargs["models"] = [model_name, *fallbacks]

    return MaskingChatModel(
        ChatOpenRouter(
            model_name=model_name,
            openrouter_api_key=s.openrouter_api_key.get_secret_value(),
            temperature=temperature,
            max_retries=2,
            timeout=60_000,
            model_kwargs=model_kwargs,
            app_url=s.openrouter_http_referer,
            app_title=s.openrouter_x_title,
        )
    )
