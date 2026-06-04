"""Unit tests for clarification agent."""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.messages import HumanMessage

from family_finance.agents.clarify import clarify_node
from family_finance.domain import Category, Direction


class FakeRepository:
    calls: ClassVar[list[dict[str, Any]]] = []

    async def classify_by_import_hashes(self, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return len(kwargs["import_hashes"])


@pytest.mark.unit
async def test_clarify_node_updates_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRepository.calls = []
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresTransactionRepository",
        FakeRepository,
    )
    family_id = uuid.uuid4()

    result = await clarify_node(
        {
            "family_id": str(family_id),
            "messages": [HumanMessage(content="1 одежда")],
            "open_questions": [
                {
                    "id": 1,
                    "reason": "непонятная категория",
                    "merchant_raw": "CONCEPT CLUB",
                    "payment_dates": ["2026-04-01"],
                    "count": 1,
                    "total": "4607.00",
                    "import_hashes": ["hash-1", "hash-2"],
                    "text": "Непонятная категория: 01.04.2026, «CONCEPT CLUB», 4607.00 ₽. Что это?",
                }
            ],
        }
    )

    assert FakeRepository.calls == [
        {
            "family_id": family_id,
            "import_hashes": ["hash-1", "hash-2"],
            "category": Category.SHOPPING_CLOTHES,
            "direction": Direction.EXPENSE,
            "subcategory_freetext": "одежда",
            "needs_review": False,
        }
    ]
    assert result["open_questions"] == []
    assert "Обновил транзакций: 2" in result["messages"][0].content
