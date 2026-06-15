"""HITL interrupt/resume cycle for ingest_node (ADR 0009).

Парсинг идёт до ``interrupt()`` (без сайд-эффектов), запись ``add_many`` — после.
Проверяем: граф встаёт на паузу до подтверждения; resume(True) пишет ровно один
раз; resume(False) отменяет и в БД не пишет.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from family_finance.agents.ingest import ingest_node
from family_finance.agents.state import FinanceState
from family_finance.domain import (
    Category,
    Currency,
    Direction,
    Transaction,
    TransactionSource,
)


def _tx() -> Transaction:
    return Transaction(
        transaction_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        amount=Decimal("100.00"),
        currency=Currency.RUB,
        direction=Direction.EXPENSE,
        merchant_raw="TEST",
        category=Category.UNCLASSIFIED,
        confidence=0.4,
        source=TransactionSource.BANK_CSV,
    )


def _build_graph() -> object:
    """Минимальный граф из одной ноды ingest, чекпоинтер в памяти."""
    graph = StateGraph(FinanceState)
    graph.add_node("ingest", ingest_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", END)
    return graph.compile(checkpointer=InMemorySaver())


def _state(csv_path: Path) -> dict[str, object]:
    return {
        "family_id": str(uuid.uuid4()),
        "member_id": str(uuid.uuid4()),
        "pending_csv": str(csv_path),
    }


@pytest.mark.unit
async def test_ingest_pauses_before_write(tmp_path: Path) -> None:
    """interrupt() встаёт ДО add_many — ничего не пишем, пока не подтвердили."""
    csv = tmp_path / "statement.csv"
    csv.write_bytes(b"dummy")
    txns = [_tx(), _tx()]
    add_many = AsyncMock(return_value=txns)

    with (
        patch("family_finance.agents.ingest.TinkoffCsvParser") as parser_cls,
        patch("family_finance.agents.ingest.PostgresTransactionRepository") as repo_cls,
    ):
        parser_cls.return_value.parse = MagicMock(return_value=txns)
        repo_cls.return_value.add_many = add_many
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-pause"}}
        result = await app.ainvoke(_state(csv), config=config)

    assert "__interrupt__" in result
    add_many.assert_not_called()  # пауза сработала — в БД ещё ничего


@pytest.mark.unit
async def test_ingest_resume_yes_writes_once(tmp_path: Path) -> None:
    """resume(True) → add_many ровно один раз, ingest_ok=True."""
    csv = tmp_path / "statement.csv"
    csv.write_bytes(b"dummy")
    txns = [_tx(), _tx()]
    add_many = AsyncMock(return_value=txns)

    with (
        patch("family_finance.agents.ingest.TinkoffCsvParser") as parser_cls,
        patch("family_finance.agents.ingest.PostgresTransactionRepository") as repo_cls,
    ):
        parser_cls.return_value.parse = MagicMock(return_value=txns)
        repo_cls.return_value.add_many = add_many
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-yes"}}
        await app.ainvoke(_state(csv), config=config)
        result = await app.ainvoke(Command(resume=True), config=config)

    add_many.assert_awaited_once()
    assert result["ingest_ok"] is True
    assert len(result["parsed_transactions"]) == 2


@pytest.mark.unit
async def test_ingest_resume_no_cancels(tmp_path: Path) -> None:
    """resume(False) → отмена, add_many НЕ вызван, pending очищен."""
    csv = tmp_path / "statement.csv"
    csv.write_bytes(b"dummy")
    txns = [_tx()]
    add_many = AsyncMock(return_value=txns)

    with (
        patch("family_finance.agents.ingest.TinkoffCsvParser") as parser_cls,
        patch("family_finance.agents.ingest.PostgresTransactionRepository") as repo_cls,
    ):
        parser_cls.return_value.parse = MagicMock(return_value=txns)
        repo_cls.return_value.add_many = add_many
        app = _build_graph()
        config = {"configurable": {"thread_id": "t-no"}}
        await app.ainvoke(_state(csv), config=config)
        result = await app.ainvoke(Command(resume=False), config=config)

    add_many.assert_not_called()
    assert result["ingest_ok"] is False
    assert result["pending_csv"] is None
    assert "отменён" in str(result["messages"][-1].content)
