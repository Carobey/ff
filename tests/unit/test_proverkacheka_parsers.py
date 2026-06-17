"""Unit tests for the pure parsers in infrastructure/parsers/proverkacheka.py."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from family_finance.infrastructure.parsers.proverkacheka import (
    _build_receipt,
    _parse_datetime,
    _parse_items,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-05-02T12:34:00", datetime(2026, 5, 2, 12, 34, 0, tzinfo=UTC)),
        ("2026-05-02T12:34", datetime(2026, 5, 2, 12, 34, tzinfo=UTC)),
        ("20260502T123456", datetime(2026, 5, 2, 12, 34, 56, tzinfo=UTC)),
        ("20260502T1234", datetime(2026, 5, 2, 12, 34, tzinfo=UTC)),
    ],
)
def test_parse_datetime_known_formats(raw: str, expected: datetime) -> None:
    """ISO and ФНС-compact strings parse to tz-aware UTC datetimes."""
    assert _parse_datetime(raw) == expected


@pytest.mark.unit
def test_parse_datetime_empty_falls_back_to_now() -> None:
    """An empty string yields a tz-aware 'now' rather than raising."""
    result = _parse_datetime("")
    assert result.tzinfo is not None


@pytest.mark.unit
def test_parse_datetime_garbage_falls_back_to_now() -> None:
    """An unparsable string degrades to now(UTC), not a crash."""
    result = _parse_datetime("не дата")
    assert result.tzinfo is not None


@pytest.mark.unit
def test_parse_items_basic_rubles() -> None:
    """A normal item (rubles, total ≤ 1M) is parsed without rescaling."""
    items = _parse_items([{"name": "Молоко", "quantity": 2, "price": "50.00", "sum": "100.00"}])
    assert len(items) == 1
    assert items[0].name == "Молоко"
    assert items[0].quantity == Decimal("2")
    assert items[0].price == Decimal("50.00")
    assert items[0].total == Decimal("100.00")


@pytest.mark.unit
def test_parse_items_kopeck_heuristic_rescales() -> None:
    """When total exceeds 1_000_000 the price/total are treated as kopecks."""
    items = _parse_items([{"name": "Ноутбук", "quantity": 1, "price": 5000000, "sum": 5000000}])
    assert items[0].price == Decimal("50000")
    assert items[0].total == Decimal("50000")


@pytest.mark.unit
def test_parse_items_skips_non_dict_and_invalid() -> None:
    """Non-dict entries and items that fail validation are silently dropped."""
    items = _parse_items(
        [
            "garbage",
            {"name": "Хлеб", "quantity": 1, "price": "30", "sum": "999"},  # total mismatch
            {"name": "Сыр", "quantity": 1, "price": "200", "sum": "200"},
        ]
    )
    assert [i.name for i in items] == ["Сыр"]


@pytest.mark.unit
def test_parse_items_non_list_returns_empty() -> None:
    """A non-list payload yields an empty list, not an error."""
    assert _parse_items({"not": "a list"}) == []


@pytest.mark.unit
def test_build_receipt_maps_fields_and_kopecks() -> None:
    """totalSum is divided by 100; fiscal fields and store name are mapped."""
    family_id = str(uuid.uuid4())
    member_id = str(uuid.uuid4())
    data: dict[str, object] = {
        "dateTime": "20260502T1234",
        "totalSum": 100000,  # kopecks → 1000.00
        "user": "ООО Ромашка",
        "fiscalDriveNumber": "FN123",
        "fiscalDocumentNumber": "42",
        "fiscalSign": "FP999",
        "items": [{"name": "Чай", "quantity": 1, "price": "1000", "sum": "1000"}],
    }

    receipt = _build_receipt(data, qr_raw="t=...&s=...", family_id=family_id, member_id=member_id)

    assert receipt.total_amount == Decimal("1000.00")
    assert receipt.store_name == "ООО Ромашка"
    assert receipt.fiscal_drive == "FN123"
    assert receipt.purchase_time == datetime(2026, 5, 2, 12, 34, tzinfo=UTC)
    assert len(receipt.items) == 1


@pytest.mark.unit
def test_build_receipt_store_name_falls_back_to_retail_place() -> None:
    """When ``user`` is absent the store name comes from ``retailPlace``."""
    receipt = _build_receipt(
        {"totalSum": 0, "retailPlace": "Пятёрочка №5", "items": []},
        qr_raw="q",
        family_id=str(uuid.uuid4()),
        member_id=str(uuid.uuid4()),
    )
    assert receipt.store_name == "Пятёрочка №5"
