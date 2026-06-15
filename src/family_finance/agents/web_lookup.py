"""Web lookup of an unknown merchant.

Triggered when the user answers «не знаю» to a clarification question instead of
picking a category. Instead of leaving the transaction unclassified, we look the
merchant up online and let the LLM categorise it from that context.

Two cheap worker calls (deliberately separate — keeps structured output off the
web-search request, where плагин подмешивает результаты в свободный текст):
  1. ``describe_merchant`` — модель с ``:online`` ищет в интернете, кто это.
  2. таксономия из справочника + описание → constrained ``CategoryPrediction``.

Любая ошибка → ``None`` / пустое описание (мягкая деградация: вернёмся к
«оставил без категории», как и раньше).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import structlog
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from family_finance.agents.categorizer import CategoryPrediction, build_system_prompt
from family_finance.domain import Category
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.persistence import PostgresCategoryCatalog

logger = structlog.get_logger()

_DESCRIBE_SYSTEM = (
    "Ты — поисковый помощник. По названию продавца из банковской выписки "
    "коротко (одно предложение, по-русски) объясни, что это за компания/"
    "заведение и чем занимается. Если не нашёл — ответь «неизвестно»."
)


@dataclass(frozen=True)
class WebVerdict:
    """Результат веб-поиска: что за продавец и какая категория ему подходит."""

    merchant_raw: str
    description: str
    category: Category


async def describe_merchant(merchant_raw: str) -> str:
    """Свободный веб-поиск: что это за продавец. Пустая строка при ошибке."""
    model = get_chat_model(tier="worker", online=True)
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=_DESCRIBE_SYSTEM),
                HumanMessage(content=f"Что это за продавец/компания: «{merchant_raw}»?"),
            ]
        )
    except Exception:
        logger.exception("web_lookup_describe_failed", merchant=merchant_raw)
        return ""
    text = str(response.content).strip()
    return "" if text.lower().startswith("неизвестно") else text


async def lookup_merchant(merchant_raw: str) -> WebVerdict | None:
    """Найти продавца в интернете и подобрать категорию из справочника.

    Возвращает ``None``, если категоризация не удалась (ошибка LLM/таксономии).
    """
    description = await describe_merchant(merchant_raw)

    try:
        taxonomy = await PostgresCategoryCatalog().render_taxonomy()
    except Exception:
        logger.exception("web_lookup_taxonomy_failed")
        taxonomy = ""

    model = cast(
        "Runnable[LanguageModelInput, CategoryPrediction]",
        get_chat_model(tier="worker").with_structured_output(CategoryPrediction),
    )
    human = f"Продавец: {merchant_raw}"
    if description:
        human += f"\nИнформация из интернета: {description}"
    try:
        prediction = await model.ainvoke(
            [SystemMessage(content=build_system_prompt(taxonomy)), HumanMessage(content=human)]
        )
    except Exception:
        logger.exception("web_lookup_categorize_failed", merchant=merchant_raw)
        return None

    return WebVerdict(
        merchant_raw=merchant_raw,
        description=description,
        category=prediction.category,
    )


async def categorize_freetext(merchant_raw: str, user_text: str) -> Category | None:
    """Категоризировать по описанию пользователя (без веб-поиска).

    Когда юзер вручную написал, что это («спортзал», «корм коту»), а ни одна
    кнопка не подошла — отдаём текст категоризатору поверх полной таксономии
    справочника. ``None`` при ошибке LLM (мягкая деградация → без категории).
    """
    try:
        taxonomy = await PostgresCategoryCatalog().render_taxonomy()
    except Exception:
        logger.exception("freetext_taxonomy_failed")
        taxonomy = ""

    model = cast(
        "Runnable[LanguageModelInput, CategoryPrediction]",
        get_chat_model(tier="worker").with_structured_output(CategoryPrediction),
    )
    human = f"Продавец: {merchant_raw}\nПользователь уточнил, что это: {user_text}"
    try:
        prediction = await model.ainvoke(
            [SystemMessage(content=build_system_prompt(taxonomy)), HumanMessage(content=human)]
        )
    except Exception:
        logger.exception("freetext_categorize_failed", merchant=merchant_raw)
        return None
    return prediction.category
