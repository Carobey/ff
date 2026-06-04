"""Unit tests for deterministic supervisor routing."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from family_finance.agents.supervisor import (
    route_after_ingest,
    route_after_supervisor,
    supervisor_node,
)


@pytest.mark.unit
def test_route_csv_to_ingest() -> None:
    assert route_after_supervisor({"pending_csv": "uploads/test.csv"}) == "ingest"


@pytest.mark.unit
def test_route_after_ingest_with_rows_to_categorizer() -> None:
    assert route_after_ingest({"ingest_ok": True}) == "categorizer"


@pytest.mark.unit
def test_route_after_ingest_empty_to_end() -> None:
    assert route_after_ingest({"ingest_ok": False}) == "end"
    assert route_after_ingest({}) == "end"


@pytest.mark.unit
def test_route_query_to_ledger() -> None:
    assert route_after_supervisor({"current_intent": "query"}) == "ledger"


@pytest.mark.unit
def test_route_clarification_to_clarify() -> None:
    assert route_after_supervisor({"current_intent": "clarify"}) == "clarify"


@pytest.mark.unit
def test_route_idle_to_end() -> None:
    assert route_after_supervisor({"current_intent": "idle"}) == "end"


@pytest.mark.unit
def test_route_subscriptions_to_subscriptions() -> None:
    assert route_after_supervisor({"current_intent": "subscriptions"}) == "subscriptions"


@pytest.mark.unit
async def test_supervisor_detects_subscriptions_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Subscriptions routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node({"messages": [HumanMessage(content="мои подписки")]})

    assert result["current_intent"] == "subscriptions"


@pytest.mark.unit
async def test_supervisor_routes_ledger_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Ledger routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node({"messages": [HumanMessage(content="сколько на аптеки?")]})

    assert result["current_intent"] == "query"


@pytest.mark.unit
async def test_supervisor_routes_clarification_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Clarification routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {
            "messages": [HumanMessage(content="1 одежда")],
            "open_questions": [
                {
                    "id": 1,
                    "reason": "непонятная категория",
                    "merchant_raw": "CONCEPT CLUB",
                    "payment_dates": ["2026-04-01"],
                    "count": 1,
                    "total": "4607.00",
                    "import_hashes": ["hash-1"],
                    "text": "Непонятная категория: 01.04.2026, «CONCEPT CLUB», 4607.00 ₽. Что это?",
                }
            ],
        }
    )

    assert result["current_intent"] == "clarify"
