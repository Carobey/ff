"""Unit tests for PII masking before outbound LLM calls."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    assert "[PHONE_NUMBER]" in masked
    assert "916" not in masked


@pytest.mark.unit
def test_mask_text_email() -> None:
    masked = mask_text("пиши на yuri.test@example.com")
    assert "[EMAIL_ADDRESS]" in masked
    assert "example.com" not in masked


@pytest.mark.unit
def test_mask_text_credit_card() -> None:
    # Luhn-valid test number — Presidio validates the checksum before masking.
    masked = mask_text("оплата картой 4111 1111 1111 1111")
    assert "[CREDIT_CARD]" in masked
    assert "4111" not in masked


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "покажи мои покупки на 30.05.2026",
        "покажи покупки 30-05-2026",
        "покажи покупки 30/05/2026",
    ],
)
def test_mask_text_does_not_treat_dates_as_phone_numbers(text: str) -> None:
    assert mask_text(text) == text


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
    assert "[PHONE_NUMBER]" in masked[1].content
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
    assert "[EMAIL_ADDRESS]" in parts[0]["text"]
    assert parts[1] == image_part  # image left untouched
