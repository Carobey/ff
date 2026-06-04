"""Natural-language parser for the weekly-digest schedule.

User types ``/digest_schedule по воскресеньям в 19:00`` (or any free-form
Russian/English variant). The handler routes that text here, we ask the
worker LLM with a structured output schema to extract ``day_of_week`` and
``time``, then return a :class:`DigestSchedule`.

If the LLM can't pick a single day + time (e.g. "каждый день" — we don't
support daily delivery in v1), we return ``None`` and the caller asks the
user to clarify.
"""

from __future__ import annotations

from typing import Literal, cast

import structlog
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from family_finance.domain import DigestSchedule
from family_finance.infrastructure.llm import get_chat_model

logger = structlog.get_logger()


class _ParsedSchedule(BaseModel):
    """LLM structured output for the schedule parser."""

    day_of_week: Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"] | None = Field(
        None, description="Один день недели. Null если пользователь не указал."
    )
    hour: int | None = Field(
        None, ge=0, le=23, description="Час в 24-часовом формате. Null если не указан."
    )
    minute: int | None = Field(0, ge=0, le=59, description="Минута, по умолчанию 0.")


_SYSTEM_PROMPT = """\
Ты — парсер расписания. Пользователь хочет получать еженедельную сводку
финансов в определённый день недели и время.

Извлеки из текста ОДИН день недели (mon/tue/wed/thu/fri/sat/sun) и время
в 24-часовом формате (hour + minute).

Правила:
- Если пользователь сказал «каждый день», «ежедневно» — верни day_of_week=null
  (мы поддерживаем только один день в неделю).
- Если не сказано время — верни hour=null.
- «утром» → hour=9, «днём» → hour=14, «вечером» → hour=19, «ночью» → hour=22.
- Если несколько дней недели — выбери первый упомянутый.
"""


async def parse_digest_schedule(text: str) -> DigestSchedule | None:
    """Parse free-form Russian into a :class:`DigestSchedule`.

    Returns ``None`` when the user input is ambiguous (e.g. no day specified
    or asking for daily delivery). Caller should reply with a clarification
    prompt in that case.
    """
    model = cast(
        "Runnable[LanguageModelInput, _ParsedSchedule]",
        get_chat_model(tier="worker").with_structured_output(_ParsedSchedule),
    )
    try:
        parsed = await model.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=text)],
        )
    except Exception:
        logger.exception("digest_schedule_parser_failed", text=text[:80])
        return None

    if parsed.day_of_week is None or parsed.hour is None:
        logger.info(
            "digest_schedule_parser_incomplete",
            text=text[:80],
            day=parsed.day_of_week,
            hour=parsed.hour,
        )
        return None

    return DigestSchedule(
        day_of_week=parsed.day_of_week,
        hour=parsed.hour,
        minute=parsed.minute or 0,
    )
