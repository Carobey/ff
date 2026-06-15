"""Unit tests for PII masking before outbound LLM calls."""

from __future__ import annotations

from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from family_finance.infrastructure.security import mask_messages, mask_text


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "Позвони мне на +7 916 123-45-67 завтра",
        "Мой телефон 8 (916) 123-45-67",
    ],
)
def test_mask_text_phone(text: str) -> None:
    masked = mask_text(text)
    assert "<PHONE_NUMBER>" in masked
    assert "916" not in masked


@pytest.mark.unit
def test_mask_text_email() -> None:
    masked = mask_text("пиши на yuri.test@example.com")
    assert "<EMAIL_ADDRESS>" in masked
    assert "example.com" not in masked


@pytest.mark.unit
def test_mask_text_credit_card() -> None:
    # Luhn-valid test number — Presidio validates the checksum before masking.
    masked = mask_text("оплата картой 4111 1111 1111 1111")
    assert "<CREDIT_CARD>" in masked
    assert "4111" not in masked


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "просто покупка продуктов в Пятёрочке на 1500 ₽",
        "Нужды: 40 000 ₽ (50%), желания: 30 000 ₽",
        "",
    ],
)
def test_mask_text_leaves_clean_text_untouched(text: str) -> None:
    assert mask_text(text) == text


@pytest.mark.unit
def test_mask_messages_only_touches_human() -> None:
    messages = [
        SystemMessage(content="Ты финансовый помощник. Телефон банка +7 495 000-00-00."),
        HumanMessage(content="мой номер +7 916 123-45-67, сколько потратил?"),
        AIMessage(content="за апрель 12 000 ₽"),
    ]
    masked = mask_messages(messages)

    # Our own system prompt is left intact; only user content is anonymized.
    assert masked[0].content == messages[0].content
    assert masked[2].content == messages[2].content
    assert "<PHONE_NUMBER>" in masked[1].content
    assert "916" not in str(masked[1].content)


@pytest.mark.unit
def test_mask_messages_handles_multimodal_content() -> None:
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    human = HumanMessage(
        content=[
            {"type": "text", "text": "вот чек, мой email yuri.test@example.com"},
            image_part,
        ]
    )
    (masked,) = mask_messages([human])
    parts = masked.content

    assert isinstance(parts, list)
    assert "<EMAIL_ADDRESS>" in parts[0]["text"]
    assert parts[1] == image_part  # image left untouched


@pytest.mark.unit
async def test_masking_survives_bind_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """PII must stay masked even when the model is used as a tool-calling ReAct agent.

    ``MaskingChatModel.bind_tools`` binds to *self*, so the bound runnable still
    routes through the masking ``_agenerate``. This regression-locks the security
    property that lets ``create_react_agent`` consume the single LLM chokepoint.
    """
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langchain_openrouter import ChatOpenRouter
    from pydantic import SecretStr

    from family_finance.infrastructure.llm import openrouter_client

    class _FakeSettings:
        llm_worker_model = "google/gemini-2.5-flash"
        llm_supervisor_model = "openai/gpt-5.4"
        worker_fallback_list: ClassVar[list[str]] = []
        supervisor_fallback_list: ClassVar[list[str]] = []
        openrouter_api_key = SecretStr("sk-test")
        openrouter_http_referer = "http://example.invalid"
        openrouter_x_title = "test"

    monkeypatch.setattr(openrouter_client, "get_settings", lambda: _FakeSettings())
    openrouter_client.get_chat_model.cache_clear()

    captured: list[list[BaseMessage]] = []

    async def fake_agenerate(
        self: ChatOpenRouter,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        captured.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    # Patch the low-level generate on the inner class — no network, no API key use.
    monkeypatch.setattr(ChatOpenRouter, "_agenerate", fake_agenerate)

    @tool
    def noop() -> str:
        """Demo tool — never actually called in this test."""
        return "x"

    model = openrouter_client.get_chat_model(tier="worker")
    bound = model.bind_tools([noop])
    await bound.ainvoke([HumanMessage(content="мой телефон +7 916 123-45-67")])
    openrouter_client.get_chat_model.cache_clear()  # don't leak the patched instance

    assert captured, "inner._agenerate must be reached through the bound runnable"
    sent = str(captured[0][-1].content)
    assert "<PHONE_NUMBER>" in sent
    assert "916" not in sent
