"""Unit tests for the CoachAgent: no-data gate, ReAct pass-through, tool render.

Coach answers behavioural questions via a ReAct loop over MCP tools; numbers
come from Python (SQL aggregates), the LLM only narrates. Tests lock the gate,
the happy/fallback paths and the deterministic tool rendering.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from family_finance.agents.coach import _build_coach_tools, coach_node

_TOTAL_EMPTY = [{"bucket": "total", "subbucket": None, "total": "0", "count": 0}]
_TOTAL_SOME = [{"bucket": "total", "subbucket": None, "total": "50000", "count": 12}]


def _coach_state(text: str) -> dict[str, object]:
    return {"messages": [HumanMessage(content=text)], "family_id": str(uuid.uuid4())}


# ── no-data gate ──────────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_coach_node_gate_no_data_skips_react() -> None:
    """Пустая история → честный ответ, ReAct-цикл не запускается."""
    with (
        patch(
            "family_finance.agents.coach.call_finance_tool",
            AsyncMock(return_value=_TOTAL_EMPTY),
        ),
        patch("family_finance.agents.coach.create_react_agent") as mock_agent,
    ):
        result = await coach_node(_coach_state("как часто я заказываю доставку?"))

    mock_agent.assert_not_called()
    assert "выписку" in str(result["messages"][0].content).lower()
    assert result["current_intent"] == "idle"


# ── ReAct happy path / fallback ───────────────────────────────────────────────


@pytest.mark.unit
async def test_coach_node_react_happy_path() -> None:
    """Есть данные → ответ ReAct-агента уходит пользователю."""

    class _FakeAgent:
        async def ainvoke(self, _inp: object, config: object | None = None) -> dict[str, object]:
            return {"messages": [AIMessage(content="Доставку заказывал 8 раз в апреле.")]}

    with (
        patch(
            "family_finance.agents.coach.call_finance_tool",
            AsyncMock(return_value=_TOTAL_SOME),
        ),
        patch("family_finance.agents.coach.get_chat_model", return_value=object()),
        patch("family_finance.agents.coach.create_react_agent", return_value=_FakeAgent()),
    ):
        result = await coach_node(_coach_state("как часто я заказываю доставку?"))

    assert "8 раз" in str(result["messages"][0].content)
    assert result["current_intent"] == "idle"


@pytest.mark.unit
async def test_coach_node_falls_back_on_agent_error() -> None:
    """ReAct падает → честный no-data ответ, не пусто и не исключение."""

    class _BoomAgent:
        async def ainvoke(self, _inp: object, config: object | None = None) -> dict[str, object]:
            raise RuntimeError("model down")

    with (
        patch(
            "family_finance.agents.coach.call_finance_tool",
            AsyncMock(return_value=_TOTAL_SOME),
        ),
        patch("family_finance.agents.coach.get_chat_model", return_value=object()),
        patch("family_finance.agents.coach.create_react_agent", return_value=_BoomAgent()),
    ):
        result = await coach_node(_coach_state("когда я последний раз так тратил?"))

    assert str(result["messages"][0].content)  # непусто
    assert result["current_intent"] == "idle"


# ── tool rendering (numbers from MCP, RU labels, counts) ──────────────────────


@pytest.mark.unit
async def test_spending_breakdown_renders_category_labels_and_counts() -> None:
    rows = [
        {"bucket": "food.delivery", "subbucket": None, "total": "12345", "count": 8},
        {"bucket": "food.groceries", "subbucket": None, "total": "40000", "count": 10},
    ]
    tools = {t.name: t for t in _build_coach_tools(uuid.uuid4())}
    with patch("family_finance.agents.coach.call_finance_tool", AsyncMock(return_value=rows)):
        out = await tools["spending_breakdown"].ainvoke(
            {"dimension": "category", "period_hint": "в апреле"}
        )

    assert "Доставка еды" in out
    assert "8 операций" in out
    assert "12 345 ₽" in out


@pytest.mark.unit
async def test_recent_transactions_renders_dates() -> None:
    rows = [
        {
            "occurred_at": "2026-04-20T12:00:00+00:00",
            "amount": "1500",
            "direction": "expense",
            "category": "food.delivery",
            "merchant": "Самокат",
        }
    ]
    tools = {t.name: t for t in _build_coach_tools(uuid.uuid4())}
    with patch("family_finance.agents.coach.call_finance_tool", AsyncMock(return_value=rows)):
        out = await tools["recent_transactions"].ainvoke({"period_hint": "", "limit": 5})

    assert "20.04.2026" in out
    assert "Самокат" in out
    assert "1 500 ₽" in out
