"""Unit tests for clarification agent."""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.messages import HumanMessage

from family_finance.agents.clarifications import (
    parse_freetext_answers,
    parse_unknown_answers,
)
from family_finance.agents.clarify import clarify_node
from family_finance.agents.web_lookup import WebVerdict
from family_finance.domain import Category, Direction


class FakeRepository:
    calls: ClassVar[list[dict[str, Any]]] = []

    async def classify_by_import_hashes(self, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return len(kwargs["import_hashes"])


class FakeRuleRepository:
    calls: ClassVar[list[dict[str, Any]]] = []

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@pytest.mark.unit
async def test_clarify_node_updates_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRepository.calls = []
    FakeRuleRepository.calls = []
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresTransactionRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresMerchantRuleRepository",
        FakeRuleRepository,
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

    # Learning loop: ответ пользователя записан как правило семьи (source=user).
    assert FakeRuleRepository.calls == [
        {
            "family_id": family_id,
            "merchant_raw": "CONCEPT CLUB",
            "category": Category.SHOPPING_CLOTHES,
            "source": "user",
        }
    ]


_QUESTION = {
    "id": 1,
    "reason": "непонятная категория",
    "merchant_raw": "СТУДИЯ ЙОГИ АУМ",
    "payment_dates": ["2026-04-01"],
    "count": 1,
    "total": "1500.00",
    "import_hashes": ["hash-1"],
    "text": "Непонятная категория: «СТУДИЯ ЙОГИ АУМ». Что это?",
}


@pytest.mark.unit
@pytest.mark.parametrize("answer", ["1 не знаю", "1 __lookup__", "1 хз"])
def test_parse_unknown_answers_detects_dont_know(answer: str) -> None:
    unknown = parse_unknown_answers(answer, [_QUESTION])
    assert [q["id"] for q in unknown] == [1]


@pytest.mark.unit
def test_parse_unknown_answers_ignores_category_pick() -> None:
    assert parse_unknown_answers("1 одежда", [_QUESTION]) == []


@pytest.mark.unit
def test_parse_freetext_answers_detects_description() -> None:
    """«1 спортзал» — не ключевое слово и не «не знаю» → ручной ввод для LLM."""
    result = parse_freetext_answers("1 спортзал", [_QUESTION])
    assert [(q["id"], label) for q, label in result] == [(1, "спортзал")]


@pytest.mark.unit
@pytest.mark.parametrize("answer", ["1 одежда", "1 не знаю"])
def test_parse_freetext_answers_skips_known_branches(answer: str) -> None:
    """Точная категория и «не знаю» обрабатываются другими ветками — не freetext."""
    assert parse_freetext_answers(answer, [_QUESTION]) == []


@pytest.mark.unit
async def test_clarify_node_freetext_categorizes_via_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Свободный текст → LLM-категоризация, правило source='llm' (категорию выбрала LLM)."""
    FakeRepository.calls = []
    FakeRuleRepository.calls = []
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresTransactionRepository", FakeRepository
    )
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresMerchantRuleRepository", FakeRuleRepository
    )

    async def fake_categorize(merchant_raw: str, user_text: str) -> Category:
        assert user_text == "спортзал"
        return Category.HEALTH_FITNESS

    monkeypatch.setattr("family_finance.agents.clarify.categorize_freetext", fake_categorize)
    family_id = uuid.uuid4()

    result = await clarify_node(
        {
            "family_id": str(family_id),
            "messages": [HumanMessage(content="1 спортзал")],
            "open_questions": [_QUESTION],
        }
    )

    assert FakeRepository.calls == [
        {
            "family_id": family_id,
            "import_hashes": ["hash-1"],
            "category": Category.HEALTH_FITNESS,
            "direction": Direction.EXPENSE,
            "subcategory_freetext": "спортзал",
            "needs_review": False,
        }
    ]
    assert FakeRuleRepository.calls == [
        {
            "family_id": family_id,
            "merchant_raw": "СТУДИЯ ЙОГИ АУМ",
            "category": Category.HEALTH_FITNESS,
            "source": "llm",
        }
    ]
    assert result["open_questions"] == []
    assert "по твоему описанию" in result["messages"][0].content


@pytest.mark.unit
async def test_clarify_node_web_lookup_on_dont_know(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Не знаю» → веб-поиск проставляет категорию и пишет правило source='llm'."""
    FakeRepository.calls = []
    FakeRuleRepository.calls = []
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresTransactionRepository", FakeRepository
    )
    monkeypatch.setattr(
        "family_finance.agents.clarify.PostgresMerchantRuleRepository", FakeRuleRepository
    )

    async def fake_lookup(merchant_raw: str) -> WebVerdict:
        return WebVerdict(
            merchant_raw=merchant_raw,
            description="студия йоги",
            category=Category.ENTERTAINMENT_HOBBIES,
        )

    monkeypatch.setattr("family_finance.agents.clarify.lookup_merchant", fake_lookup)
    family_id = uuid.uuid4()

    result = await clarify_node(
        {
            "family_id": str(family_id),
            "messages": [HumanMessage(content="1 не знаю")],
            "open_questions": [_QUESTION],
        }
    )

    # Транзакция категоризирована найденной из интернета категорией.
    assert FakeRepository.calls == [
        {
            "family_id": family_id,
            "import_hashes": ["hash-1"],
            "category": Category.ENTERTAINMENT_HOBBIES,
            "direction": Direction.EXPENSE,
            "needs_review": False,
        }
    ]
    # Правило записано как машинная догадка (source=llm), не подтверждённая юзером.
    assert FakeRuleRepository.calls == [
        {
            "family_id": family_id,
            "merchant_raw": "СТУДИЯ ЙОГИ АУМ",
            "category": Category.ENTERTAINMENT_HOBBIES,
            "source": "llm",
        }
    ]
    assert result["open_questions"] == []
    assert "Нашёл в интернете" in result["messages"][0].content
