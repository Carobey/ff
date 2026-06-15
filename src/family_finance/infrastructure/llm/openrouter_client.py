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
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
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


class MaskingChatModel(BaseChatModel):
    """Chat model that masks PII on every outbound call.

    This is the single LLM chokepoint (CLAUDE.md «Адаптер ОДИН»): every node's
    prompt is anonymized before it reaches OpenRouter without any per-node
    change. It is a real :class:`BaseChatModel` so that ``create_react_agent``
    (and any other LangGraph helper) can ``bind_tools`` against it — the masking
    lives in ``_agenerate``/``_generate``, so it stays inside the tool-calling
    loop. ``bind_tools`` binds to *self* (not ``inner``) precisely so the bound
    runnable still routes back through the masking generate methods.
    """

    inner: ChatOpenRouter

    @property
    def _llm_type(self) -> str:
        return "masking-openrouter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.inner._generate(mask_messages(messages), stop, run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self.inner._agenerate(mask_messages(messages), stop, run_manager, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: dict[str, Any] | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Format tools and bind them to *self* so masking stays in the loop.

        Mirrors ``ChatOpenRouter.bind_tools`` but calls ``self.bind`` instead of
        ``super().bind`` — the resulting binding wraps this masking model, so
        each ReAct step still anonymizes the prompt before the HTTP call.
        """
        formatted_tools = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. "
                        f"Received {len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {"type": "function", "function": {"name": tool_name}}
            kwargs["tool_choice"] = tool_choice
        return cast(
            "Runnable[LanguageModelInput, AIMessage]",
            self.bind(tools=formatted_tools, **kwargs),
        )

    # The generic facade return type is more precise than BaseChatModel's
    # ``Runnable[..., dict | BaseModel]`` — callers rely on the per-schema type.
    def with_structured_output[T](  # type: ignore[override]
        self, schema: type[T], **kwargs: Any
    ) -> _MaskingRunnable[T]:
        inner = cast(
            "Runnable[LanguageModelInput, T]",
            self.inner.with_structured_output(cast("Any", schema), **kwargs),
        )
        return _MaskingRunnable(inner)


@lru_cache(maxsize=8)
def get_chat_model(
    *, tier: str = "worker", temperature: float = 0.1, online: bool = False
) -> MaskingChatModel:
    """
    Получить chat-модель через OpenRouter. Кешируется по tier+temperature+online.

    Возвращает обёртку :class:`MaskingChatModel` — она маскирует PII в тексте
    пользователя перед отправкой в облачный LLM (см. infrastructure/security).

    tier:
        "supervisor" — primary: gpt-5.4 (с fallbacks)
        "worker"     — primary: gemini-2.5-flash (с fallbacks)
    online:
        True — добавляет суффикс ``:online`` к модели (и fallbacks): OpenRouter
        подмешивает результаты веб-поиска в контекст. Платно (~$0.02/запрос),
        поэтому включаем точечно — например, при ответе пользователя «не знаю».
        https://openrouter.ai/docs/features/web-search
    """
    s = get_settings()

    if tier == "supervisor":
        model_name = s.llm_supervisor_model
        fallbacks = s.supervisor_fallback_list
    else:
        model_name = s.llm_worker_model
        fallbacks = s.worker_fallback_list

    if online:
        model_name = f"{model_name}:online"
        fallbacks = [f"{f}:online" for f in fallbacks]

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
        inner=ChatOpenRouter(
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
