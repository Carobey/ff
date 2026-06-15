"""HITL interrupt/resume для tax_node (ст. 219 НК).

Чтение трат/дохода идёт ДО ``interrupt()`` (без сайд-эффектов), расчёт — ПОСЛЕ.
Проверяем: пауза, когда нужны флаги/доход; resume считает возврат; нет вычитаемых
трат — прямой ответ без паузы; нечего спрашивать (только спорт + доход виден) —
тоже без паузы.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from family_finance.agents.state import FinanceState
from family_finance.agents.tax import tax_node
from family_finance.application.ports import LedgerSummary
from family_finance.domain import Category


def _build_graph() -> object:
    graph = StateGraph(FinanceState)
    graph.add_node("tax", tax_node)
    graph.add_edge(START, "tax")
    graph.add_edge("tax", END)
    return graph.compile(checkpointer=InMemorySaver())


def _state() -> dict[str, object]:
    return {"family_id": str(uuid.uuid4()), "member_id": str(uuid.uuid4())}


def _reader(
    breakdown: list[tuple[Category, Decimal, int]],
    income: Decimal,
) -> AsyncMock:
    inst = AsyncMock()
    inst.category_breakdown = AsyncMock(return_value=breakdown)
    inst.aggregate = AsyncMock(return_value=LedgerSummary(total=income, count=12))
    return inst


@pytest.mark.unit
async def test_tax_pauses_to_ask_flags() -> None:
    """Есть медицина → нужен флаг «дорогостоящее» → interrupt до расчёта."""
    breakdown = [
        (Category.TAX_DED_MEDICAL, Decimal("100000"), 3),
        (Category.TAX_DED_SPORT, Decimal("20000"), 1),
    ]
    with patch("family_finance.agents.tax.MCPLedgerReader") as reader_cls:
        reader_cls.return_value = _reader(breakdown, Decimal("1200000"))
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-pause"}}
        result = await app.ainvoke(_state(), config=config)

    assert "__interrupt__" in result


@pytest.mark.unit
async def test_tax_resume_computes_refund() -> None:
    """resume с ответами → детерминированный возврат в сообщении."""
    breakdown = [
        (Category.TAX_DED_MEDICAL, Decimal("100000"), 3),
        (Category.TAX_DED_SPORT, Decimal("20000"), 1),
    ]
    with patch("family_finance.agents.tax.MCPLedgerReader") as reader_cls:
        reader_cls.return_value = _reader(breakdown, Decimal("1200000"))
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-resume"}}
        await app.ainvoke(_state(), config=config)
        answers = {"medical_expensive": "0", "education_children": "0", "children_count": 0}
        result = await app.ainvoke(Command(resume=answers), config=config)

    # general = medical 100000 + sport 20000 = 120000; 13% × 120000 = 15600
    assert "15 600 ₽" in str(result["messages"][-1].content)
    assert result["current_intent"] == "idle"


@pytest.mark.unit
async def test_tax_no_deductible_spend_answers_directly() -> None:
    """Нет вычитаемых трат → прямой ответ, без паузы."""
    breakdown = [(Category.FOOD_GROCERIES, Decimal("50000"), 10)]
    with patch("family_finance.agents.tax.MCPLedgerReader") as reader_cls:
        reader_cls.return_value = _reader(breakdown, Decimal("1200000"))
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-empty"}}
        result = await app.ainvoke(_state(), config=config)

    assert "__interrupt__" not in result
    assert "социальный вычет" in str(result["messages"][-1].content)


@pytest.mark.unit
async def test_tax_no_questions_computes_without_pause() -> None:
    """Только спорт + доход виден → спрашивать нечего → считаем сразу."""
    breakdown = [(Category.TAX_DED_SPORT, Decimal("20000"), 1)]
    with patch("family_finance.agents.tax.MCPLedgerReader") as reader_cls:
        reader_cls.return_value = _reader(breakdown, Decimal("1200000"))
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-direct"}}
        result = await app.ainvoke(_state(), config=config)

    assert "__interrupt__" not in result
    # 13% × 20000 = 2600
    assert "2 600 ₽" in str(result["messages"][-1].content)
