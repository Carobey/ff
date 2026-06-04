"""Vision LLM fallback: extract fiscal receipt data from photo when QR decode fails.

Sends the image to a vision-capable worker model via the OpenRouter gateway
(``get_chat_model``) and asks it to extract fiscal fields visible in the
receipt header (ФН, ФД, ФП, date, total amount).

Returns a dict compatible with ``parse_fiscal_qr`` output::

    {'fn': ..., 'fp': ..., 'i': ..., 't': ..., 's': ...}

or ``None`` if extraction failed.
"""

from __future__ import annotations

import base64
import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from family_finance.infrastructure.llm import get_chat_model

logger = structlog.get_logger()


class FiscalFields(BaseModel):
    """Structured output schema for fiscal receipt fields.

    All fields are optional because the vision model may not see every field
    on a low-quality photo. The caller checks that the minimum subset
    (fn + fp + i) is present before continuing.
    """

    fn: str | None = Field(None, description="ФН (фискальный накопитель), 16 цифр")
    fp: str | None = Field(None, description="ФП (фискальный признак), 10 цифр")
    i: str | None = Field(None, description="ФД (номер фискального документа)")
    t: str | None = Field(
        None,
        description="Дата/время: ДД.ММ.ГГГГ ЧЧ:ММ или YYYYMMDDTHHMMSS",
    )
    s: str | None = Field(None, description="Итоговая сумма, только число")

    @field_validator("t")
    @classmethod
    def normalize_datetime(cls, raw: str | None) -> str | None:
        return _normalize_datetime(raw) if raw else raw


_SYSTEM_PROMPT = """\
Ты — OCR-ассистент. На фото кассовый чек.
Извлеки следующие поля из текста чека (НЕ из QR-кода):
- fn: ФН (фискальный накопитель), 16 цифр
- fp: ФП (фискальный признак), 10 цифр
- i:  ФД (номер фискального документа), число
- t:  дата и время в формате ДД.ММ.ГГГГ ЧЧ:ММ
- s:  итоговая сумма (только число, без валюты)

Если поле не видно — оставь его пустым (null).
"""


async def extract_fiscal_from_image(image_bytes: bytes) -> dict[str, str] | None:
    """Send receipt photo to worker LLM and extract fiscal fields.

    Returns dict with subset of ``{fn, fp, i, t, s}`` when ``fn + fp + i`` are
    all present; ``None`` otherwise (caller falls back to a user-facing error).
    """
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    model = get_chat_model(tier="worker").with_structured_output(FiscalFields)
    user_message = HumanMessage(
        content=[
            {"type": "text", "text": "Извлеки фискальные данные из чека."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    )

    try:
        fields = await model.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), user_message],
        )
    except Exception:
        logger.exception("vision_receipt: LLM call failed")
        return None

    data = {k: v for k, v in fields.model_dump().items() if v}
    if not {"fn", "fp", "i"}.issubset(data.keys()):
        logger.warning("vision_receipt: fiscal subset missing", fields=list(data.keys()))
        return None

    logger.info("vision_receipt: extracted fiscal data", fields=list(data.keys()))
    return data


def _normalize_datetime(raw: str) -> str:
    """Convert ``DD.MM.YYYY HH:MM`` → ``YYYYMMDDTHHMMSS`` for ProverkaCheka."""
    raw = raw.strip()
    if re.match(r"^\d{8}T\d{4,6}$", raw):
        return raw
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})(?::(\d{2}))?", raw)
    if m:
        d, mo, y, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        ss = m.group(6) or "00"
        return f"{y}{mo}{d}T{hh}{mm}{ss}"
    return raw
