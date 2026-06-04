"""Telegram text helpers."""

from __future__ import annotations

from aiogram.types import Message


async def answer_plain(message: Message, text: object) -> None:
    """Send dynamic text without Telegram HTML parsing."""
    await message.answer(str(text), parse_mode=None)
