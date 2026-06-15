"""Unit tests for deterministic ledger query parsing."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from family_finance.agents import ledger as ledger_module
from family_finance.agents.ledger import (
    QueryShape,
    _bucket_label,
    _fmt_money,
    _render_grouped,
    _render_grouped_2d,
    is_ledger_question,
    ledger_node,
    parse_ledger_query,
)
from family_finance.domain import Category, Direction


@pytest.mark.unit
def test_parse_supermarket_query_for_may() -> None:
    query = parse_ledger_query("Сколько на супермаркеты в мае?", current_year=2026)

    assert query is not None
    assert query.label == "супермаркеты"
    assert query.categories == (Category.FOOD_GROCERIES,)
    assert query.directions == (Direction.EXPENSE,)
    assert query.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert query.end == datetime(2026, 6, 1, tzinfo=UTC)


@pytest.mark.unit
def test_parse_food_query_includes_food_categories() -> None:
    query = parse_ledger_query("траты на еду", current_year=2026)

    assert query is not None
    assert query.categories == (
        Category.FOOD_GROCERIES,
        Category.FOOD_RESTAURANT,
        Category.FOOD_DELIVERY,
    )
    assert query.period_label == "за все время"


@pytest.mark.unit
def test_parse_transfer_query_uses_transfer_direction() -> None:
    query = parse_ledger_query("сколько переводов", current_year=2026)

    assert query is not None
    assert query.categories == (Category.TRANSFER_INTERNAL,)
    assert query.directions == (Direction.TRANSFER,)


@pytest.mark.unit
def test_parse_current_month_query() -> None:
    query = parse_ledger_query(
        "сколько потратил на аптеки за этот месяц",
        now=datetime(2026, 5, 28, tzinfo=UTC),
    )

    assert query is not None
    assert query.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert query.end == datetime(2026, 6, 1, tzinfo=UTC)
    assert query.period_label == "за 05.2026"


@pytest.mark.unit
def test_parse_previous_month_query() -> None:
    query = parse_ledger_query(
        "траты на еду за прошлый месяц",
        now=datetime(2026, 5, 28, tzinfo=UTC),
    )

    assert query is not None
    assert query.start == datetime(2026, 4, 1, tzinfo=UTC)
    assert query.end == datetime(2026, 5, 1, tzinfo=UTC)
    assert query.period_label == "за 04.2026"


@pytest.mark.unit
def test_parse_day_range_query() -> None:
    query = parse_ledger_query("сколько на аптеки с 1 по 15 апреля", current_year=2026)

    assert query is not None
    assert query.start == datetime(2026, 4, 1, tzinfo=UTC)
    assert query.end == datetime(2026, 4, 16, tzinfo=UTC)
    assert query.period_label == "с 01.04.2026 по 15.04.2026"


@pytest.mark.unit
def test_unknown_category_is_not_parsed() -> None:
    assert parse_ledger_query("сколько на неизвестное", current_year=2026) is None


@pytest.mark.unit
def test_ledger_question_detection() -> None:
    assert is_ledger_question("сколько на аптеки?")
    assert is_ledger_question("траты на заправки")
    assert not is_ledger_question("привет")


@pytest.mark.unit
def test_fmt_money_uses_space_thousands_separator() -> None:
    assert _fmt_money(1000) == "1 000 ₽"
    assert _fmt_money(34491) == "34 491 ₽"


@pytest.mark.unit
def test_bucket_label_renders_russian_day_and_category() -> None:
    assert _bucket_label("day", "2026-04-01") == "1 апреля"
    assert _bucket_label("month", "2026-04") == "Апрель 2026"
    assert _bucket_label("category", "food.groceries") == "Продукты"
    # Unknown category code degrades gracefully to the raw value.
    assert _bucket_label("category", "weird.code") == "weird.code"


@pytest.mark.unit
def test_render_grouped_total_equals_sum_of_rows() -> None:
    """Printed «Итого» must equal the sum of the printed per-bucket rows."""
    query = parse_ledger_query("траты за апрель", current_year=2026)
    assert query is not None
    buckets = [
        ("2026-04-01", Decimal("1000.00"), 1),
        ("2026-04-02", Decimal("1500.00"), 1),
        ("2026-04-03", Decimal("2000.00"), 2),
    ]
    rendered = _render_grouped(query, "day", buckets)

    assert "1 апреля: 1 000 ₽" in rendered
    assert "Итого: 4 500 ₽" in rendered  # 1000 + 1500 + 2000, adds up exactly


@pytest.mark.unit
def test_unclassified_is_counted_in_total_expenses() -> None:
    """«Все расходы» должны включать UNCLASSIFIED (требование Юрия)."""
    query = parse_ledger_query("сколько потратил за апрель", current_year=2026)

    assert query is not None
    assert Category.UNCLASSIFIED in query.categories


@pytest.mark.unit
def test_render_grouped_2d_subtotals_and_total_add_up() -> None:
    """2-D разбивка: подытоги по бакетам и общий «Итого» считаются в Python."""
    query = parse_ledger_query("траты за апрель", current_year=2026)
    assert query is not None
    rows = [
        ("2026-04-01", "food.groceries", Decimal("2000.00")),
        ("2026-04-01", "health.pharmacy", Decimal("1000.00")),
        ("2026-04-02", "food.delivery", Decimal("1500.00")),
    ]
    rendered = _render_grouped_2d(query, "day", "category", rows)

    assert "1 апреля — 3 000 ₽:" in rendered
    assert "• Продукты: 2 000 ₽" in rendered
    assert "• Аптека: 1 000 ₽" in rendered
    assert "2 апреля — 1 500 ₽:" in rendered
    assert "Итого: 4 500 ₽" in rendered  # 2000 + 1000 + 1500


@pytest.mark.unit
async def test_ledger_node_empty_total_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """count=0 → детерминированное «не найдено», без LLM-сравнения «-100%»."""

    async def fake_shape(_question: str) -> QueryShape:
        return QueryShape(mode="aggregate", group_by="total")

    async def fake_call(name: str, arguments: dict[str, Any]) -> Any:
        return [{"bucket": "total", "total": "0", "count": 0}]

    def fail_narrative(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("narrative LLM must not run when count==0")

    monkeypatch.setattr(ledger_module, "_extract_query_shape", fake_shape)
    monkeypatch.setattr(ledger_module, "call_finance_tool", fake_call)
    monkeypatch.setattr(ledger_module, "_narrative_llm", fail_narrative)

    result = await ledger_node(
        {
            "family_id": str(uuid.uuid4()),
            "messages": [HumanMessage(content="сколько потратил на аптеки в мае")],
        }
    )

    content = str(result["messages"][0].content)
    assert "не найдено" in content
    assert "%" not in content


@pytest.mark.unit
async def test_ledger_node_2d_breakdown_uses_subbucket_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """group_by=day + then_by=category → 2-D рендер из MCP-строк с subbucket."""

    async def fake_shape(_question: str) -> QueryShape:
        return QueryShape(mode="aggregate", group_by="day", then_by="category")

    captured: dict[str, Any] = {}

    async def fake_call(name: str, arguments: dict[str, Any]) -> Any:
        captured["arguments"] = arguments
        return [
            {"bucket": "2026-04-01", "subbucket": "food.groceries", "total": "2000.00", "count": 2},
            {
                "bucket": "2026-04-01",
                "subbucket": "health.pharmacy",
                "total": "1000.00",
                "count": 1,
            },
        ]

    monkeypatch.setattr(ledger_module, "_extract_query_shape", fake_shape)
    monkeypatch.setattr(ledger_module, "call_finance_tool", fake_call)

    result = await ledger_node(
        {
            "family_id": str(uuid.uuid4()),
            "messages": [HumanMessage(content="траты по дням по категориям за апрель")],
        }
    )

    assert captured["arguments"]["group_by"] == "day"
    assert captured["arguments"]["then_by"] == "category"
    content = str(result["messages"][0].content)
    assert "1 апреля — 3 000 ₽:" in content
    assert "• Продукты: 2 000 ₽" in content
    assert "Итого: 3 000 ₽" in content


@pytest.mark.unit
async def test_ledger_node_day_breakdown_uses_real_mcp_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """group_by=day → numbers come from MCP rows, rendered deterministically."""

    async def fake_shape(_question: str) -> QueryShape:
        return QueryShape(mode="aggregate", group_by="day")

    captured: dict[str, Any] = {}

    async def fake_call(name: str, arguments: dict[str, Any]) -> Any:
        captured["name"] = name
        captured["arguments"] = arguments
        return [
            {"bucket": "2026-04-01", "total": "1000.00", "count": 1},
            {"bucket": "2026-04-02", "total": "1500.00", "count": 1},
        ]

    monkeypatch.setattr(ledger_module, "_extract_query_shape", fake_shape)
    monkeypatch.setattr(ledger_module, "call_finance_tool", fake_call)

    result = await ledger_node(
        {
            "family_id": str(uuid.uuid4()),
            "messages": [HumanMessage(content="покажи траты по дням за апрель")],
        }
    )

    assert captured["name"] == "query_aggregates"
    assert captured["arguments"]["group_by"] == "day"
    content = str(result["messages"][0].content)
    assert "1 апреля: 1 000 ₽" in content
    assert "2 апреля: 1 500 ₽" in content
    assert "Итого: 2 500 ₽" in content  # real sum, not a hallucinated total
