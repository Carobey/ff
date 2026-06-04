"""Unit tests for Telegram text helpers."""

from __future__ import annotations

import pytest

from family_finance.bot.telegram_text import answer_plain


class _FakeMessage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def answer(self, text: str, **kwargs: object) -> None:
        self.calls.append({"text": text, **kwargs})


@pytest.mark.unit
async def test_answer_plain_disables_html_parse_mode() -> None:
    message = _FakeMessage()

    await answer_plain(message, "номер [PHONE_NUMBER] и <phone_number>")

    assert message.calls == [
        {"text": "номер [PHONE_NUMBER] и <phone_number>", "parse_mode": None}
    ]
