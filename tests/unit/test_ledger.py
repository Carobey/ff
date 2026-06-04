"""Unit tests for deterministic ledger query parsing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from family_finance.agents.ledger import is_ledger_question, parse_ledger_query
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
