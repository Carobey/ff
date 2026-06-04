"""Unit tests for SberPdfParser."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from family_finance.domain import Category, Direction, TransactionSource
from family_finance.infrastructure.parsers.sber_pdf import SberPdfParser

_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


# ── _parse_amount ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_amount_expense() -> None:
    amount, is_positive = SberPdfParser._parse_amount("27 900,00")
    assert amount == Decimal("27900.00")
    assert is_positive is False


@pytest.mark.unit
def test_parse_amount_income_with_plus() -> None:
    amount, is_positive = SberPdfParser._parse_amount("+14 000,00")
    assert amount == Decimal("14000.00")
    assert is_positive is True


@pytest.mark.unit
def test_parse_amount_small() -> None:
    amount, is_positive = SberPdfParser._parse_amount("382,00")
    assert amount == Decimal("382.00")
    assert is_positive is False


# ── _classify ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_classify_supermarket() -> None:
    direction, category = SberPdfParser._classify("Супермаркеты", False)
    assert direction == Direction.EXPENSE
    assert category == Category.FOOD_GROCERIES


@pytest.mark.unit
def test_classify_transfer_incoming() -> None:
    """Перевод с + → INCOME (деньги пришли от другого человека)."""
    direction, category = SberPdfParser._classify("Перевод на карту", True)
    assert direction == Direction.INCOME
    assert category == Category.INCOME_OTHER


@pytest.mark.unit
def test_classify_transfer_outgoing() -> None:
    """Перевод без + → TRANSFER (уходит, внутренний)."""
    direction, category = SberPdfParser._classify("Перевод СБП", False)
    assert direction == Direction.TRANSFER
    assert category == Category.TRANSFER_INTERNAL


@pytest.mark.unit
def test_classify_unknown_expense() -> None:
    """Неизвестная категория без + → UNCLASSIFIED расход."""
    direction, category = SberPdfParser._classify("Прочие операции", False)
    assert direction == Direction.EXPENSE
    assert category == Category.UNCLASSIFIED


@pytest.mark.unit
def test_classify_unknown_positive() -> None:
    """Неизвестная категория с + → INCOME."""
    direction, category = SberPdfParser._classify("Прочие операции", True)
    assert direction == Direction.INCOME
    assert category == Category.INCOME_OTHER


# ── _parse_transactions ──────────────────────────────────────────────────────

_SAMPLE_SECTION = """\
и код авторизации
      23.05.2026 09:48 Прочие операции                                  27 900,00
      23.05.2026 224085 SBERBANK ONL@IN KARTA-VKLAD. Операция по карте ****9206
      23.05.2026 09:47 Перевод СБП                                      +14 000,00
      23.05.2026 659634 Перевод от Ц. Юрий Владимирович. Операция по карте ****9206
Дата формирования документа 29.05.2026
"""


@pytest.mark.unit
def test_parse_transactions_basic() -> None:
    parser = SberPdfParser()
    rows = parser._parse_transactions(_SAMPLE_SECTION)
    assert len(rows) == 2

    r0 = rows[0]
    assert r0["date_str"] == "23.05.2026"
    assert r0["time_str"] == "09:48"
    assert r0["sber_category"] == "Прочие операции"
    assert r0["amount_raw"] == "27 900,00"
    assert "Операция по карте" not in r0["merchant"]
    assert "SBERBANK" in r0["merchant"]


@pytest.mark.unit
def test_parse_transactions_skips_non_tx_lines() -> None:
    """Строки без паттерна даты+времени не попадают в результат."""
    parser = SberPdfParser()
    text = "и код авторизации\nПроизвольный текст\nДата формирования документа"
    rows = parser._parse_transactions(text)
    assert rows == []


# ── Integration: real PDF ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_real_sber_pdf() -> None:
    """Parse the anonymized Sberbank PDF sample and verify basic structure."""
    from pathlib import Path

    pdf_path = Path(__file__).parent.parent / "samples" / "sber_sample.pdf"
    if not pdf_path.exists():
        pytest.skip("Sber PDF sample not found")

    content = pdf_path.read_bytes()

    transactions = SberPdfParser().parse(content, family_id=_FAMILY_ID, member_id=_MEMBER_ID)

    assert len(transactions) > 0, "Should parse at least one transaction"

    # All amounts must be positive
    for tx in transactions:
        assert tx.amount > 0, f"amount must be positive, got {tx.amount}"

    # Source must be BANK_PDF
    assert all(tx.source == TransactionSource.BANK_PDF for tx in transactions)

    # Check known transactions from the PDF
    amounts = {tx.amount for tx in transactions}
    assert Decimal("27900.00") in amounts, "Expense 27900 should be parsed"
    assert Decimal("14000.00") in amounts, "Income 14000 should be parsed"
    assert Decimal("599.00") in amounts, "IVI.RU 599 should be parsed"

    # Incoming transfers should be INCOME
    incoming = [tx for tx in transactions if tx.amount == Decimal("14000.00")]
    assert incoming[0].direction == Direction.INCOME

    # Entertainment should map correctly
    ivi = [tx for tx in transactions if "IVI" in tx.merchant_raw]
    if ivi:
        assert ivi[0].category == Category.ENTERTAINMENT_EVENTS
