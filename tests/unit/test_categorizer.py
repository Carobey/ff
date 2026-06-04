"""Unit tests for CategorizerAgent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from family_finance.agents.categorizer import (
    CategoryPrediction,
    _merge_enriched,
    _select_for_llm,
    categorizer_node,
)
from family_finance.domain import Category, Direction, Transaction, TransactionSource


def _make_tx(
    merchant: str = "Unknown Shop",
    category: Category = Category.UNCLASSIFIED,
    direction: Direction = Direction.EXPENSE,
    confidence: float = 0.0,
    needs_review: bool = True,
) -> Transaction:
    return Transaction(
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC),
        amount=Decimal("500.00"),
        currency="RUB",  # type: ignore[arg-type]
        direction=direction,
        merchant_raw=merchant,
        category=category,
        confidence=confidence,
        source=TransactionSource.BANK_CSV,
        import_hash="abc123",
    )


# ── _select_for_llm ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_select_for_llm_unclassified_only() -> None:
    txs = [
        _make_tx(category=Category.UNCLASSIFIED),
        _make_tx(category=Category.FOOD_GROCERIES, confidence=0.8, needs_review=False),
        _make_tx(category=Category.SHOPPING_CLOTHES, confidence=0.5, needs_review=True),
    ]
    selected = _select_for_llm(txs, all_=False)
    assert len(selected) == 2  # UNCLASSIFIED + needs_review


@pytest.mark.unit
def test_select_for_llm_all_skips_transfers() -> None:
    txs = [
        _make_tx(category=Category.FOOD_GROCERIES, direction=Direction.EXPENSE),
        _make_tx(category=Category.TRANSFER_INTERNAL, direction=Direction.TRANSFER),
    ]
    selected = _select_for_llm(txs, all_=True)
    assert len(selected) == 1
    assert selected[0].direction == Direction.EXPENSE


# ── _merge_enriched ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_merge_enriched_replaces_matched() -> None:
    original = _make_tx(category=Category.UNCLASSIFIED)
    enriched_tx = original.model_copy(
        update={"category": Category.FOOD_GROCERIES, "confidence": 0.9}
    )
    merged = _merge_enriched([original], {"abc123": enriched_tx})
    assert len(merged) == 1
    assert merged[0].category == Category.FOOD_GROCERIES


@pytest.mark.unit
def test_merge_enriched_keeps_unmatched() -> None:
    tx_keep = Transaction(
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC),
        amount=Decimal("100.00"),
        currency="RUB",  # type: ignore[arg-type]
        direction=Direction.EXPENSE,
        merchant_raw="KeepMe",
        category=Category.FOOD_RESTAURANT,
        confidence=0.9,
        source=TransactionSource.BANK_CSV,
        import_hash="keep_hash",
    )
    tx_enrich = Transaction(
        family_id=uuid.uuid4(),
        member_id=uuid.uuid4(),
        occurred_at=datetime(2026, 5, 1, 11, 0, 0, tzinfo=UTC),
        amount=Decimal("200.00"),
        currency="RUB",  # type: ignore[arg-type]
        direction=Direction.EXPENSE,
        merchant_raw="Unknown",
        category=Category.UNCLASSIFIED,
        confidence=0.0,
        source=TransactionSource.BANK_CSV,
        import_hash="enrich_hash",
    )
    enriched_version = tx_enrich.model_copy(update={"category": Category.HEALTH_PHARMACY})
    merged = _merge_enriched([tx_keep, tx_enrich], {"enrich_hash": enriched_version})
    assert len(merged) == 2
    keep_result = next(t for t in merged if t.import_hash == "keep_hash")
    assert keep_result.category == Category.FOOD_RESTAURANT  # unchanged


# ── categorizer_node ──────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_categorizer_node_enriches_unclassified() -> None:
    tx = _make_tx(merchant="ПЯТЕРОЧКА 4587", category=Category.UNCLASSIFIED)
    state = {
        "family_id": str(uuid.uuid4()),
        "parsed_transactions": [tx],
    }

    mock_prediction = CategoryPrediction(
        category=Category.FOOD_GROCERIES,
        confidence=0.95,
        reasoning="Пятёрочка — продуктовый супермаркет",
    )

    with (
        patch("family_finance.agents.categorizer.get_settings") as mock_settings,
        patch("family_finance.agents.categorizer.get_chat_model") as mock_llm,
        patch("family_finance.agents.categorizer.PostgresTransactionRepository") as mock_repo_cls,
    ):
        settings_obj = MagicMock()
        settings_obj.llm_categorize_all = False
        mock_settings.return_value = settings_obj

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=mock_prediction)
        mock_llm.return_value = mock_model

        repo_instance = MagicMock()
        repo_instance.classify_by_import_hashes = AsyncMock(return_value=1)
        mock_repo_cls.return_value = repo_instance

        result = await categorizer_node(state)  # type: ignore[arg-type]

    assert "parsed_transactions" in result
    enriched = result["parsed_transactions"]
    assert isinstance(enriched, list)
    assert enriched[0].category == Category.FOOD_GROCERIES
    assert enriched[0].confidence == 0.95
    assert enriched[0].needs_review is False


@pytest.mark.unit
async def test_categorizer_node_skips_empty_state() -> None:
    state = {"family_id": str(uuid.uuid4()), "parsed_transactions": []}
    result = await categorizer_node(state)  # type: ignore[arg-type]
    # No messages, no LLM calls
    assert result.get("messages") is None or result.get("messages") == []


@pytest.mark.unit
async def test_categorizer_node_handles_llm_error_gracefully() -> None:
    tx = _make_tx(category=Category.UNCLASSIFIED)
    state = {
        "family_id": str(uuid.uuid4()),
        "parsed_transactions": [tx],
    }

    with (
        patch("family_finance.agents.categorizer.get_settings") as mock_settings,
        patch("family_finance.agents.categorizer.get_chat_model") as mock_llm,
        patch("family_finance.agents.categorizer.PostgresTransactionRepository") as mock_repo_cls,
    ):
        settings_obj = MagicMock()
        settings_obj.llm_categorize_all = False
        mock_settings.return_value = settings_obj

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_model
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_llm.return_value = mock_model

        repo_instance = MagicMock()
        repo_instance.classify_by_import_hashes = AsyncMock(return_value=1)
        mock_repo_cls.return_value = repo_instance

        result = await categorizer_node(state)  # type: ignore[arg-type]

    # Should return original transaction unchanged (not crash)
    enriched = result.get("parsed_transactions") or []
    assert len(enriched) == 1
    assert enriched[0].category == Category.UNCLASSIFIED
