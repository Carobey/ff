"""Unit tests for QR decoder and fiscal QR parser."""

from __future__ import annotations

import pytest

from family_finance.infrastructure.parsers.qr_decoder import (
    decode_qr,
    parse_fiscal_qr,
)

# ── parse_fiscal_qr ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_fiscal_qr_standard() -> None:
    qr = "t=20260502T123456&s=1234.56&fn=9289000100123456&i=12345&fp=1234567890&n=1"
    result = parse_fiscal_qr(qr)
    assert result["t"] == "20260502T123456"
    assert result["s"] == "1234.56"
    assert result["fn"] == "9289000100123456"
    assert result["fp"] == "1234567890"
    assert result["n"] == "1"


@pytest.mark.unit
def test_parse_fiscal_qr_with_url_prefix() -> None:
    qr = "https://proverkacheka.com/check?t=20260502T1234&s=99.00&fn=111&i=222&fp=333&n=1"
    result = parse_fiscal_qr(qr)
    assert result["t"] == "20260502T1234"
    assert result["fn"] == "111"
    assert result["fp"] == "333"


@pytest.mark.unit
def test_parse_fiscal_qr_empty_returns_empty_dict() -> None:
    assert parse_fiscal_qr("") == {}


@pytest.mark.unit
def test_parse_fiscal_qr_no_equals_skipped() -> None:
    result = parse_fiscal_qr("noequals&t=20260101T0000&fn=1&fp=1")
    assert "t" in result
    assert "noequals" not in result


# ── decode_qr ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_decode_qr_invalid_bytes_returns_none() -> None:
    """Random bytes that are not an image → None, no exception."""
    result = decode_qr(b"this is not an image")
    assert result is None


@pytest.mark.unit
def test_decode_qr_empty_bytes_returns_none() -> None:
    result = decode_qr(b"")
    assert result is None
