"""Unit tests for the ProverkaCheka API adapter."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from family_finance.infrastructure.parsers import proverkacheka
from family_finance.infrastructure.parsers.proverkacheka import (
    ProverkaCheckaClient,
    ProverkaCheckaError,
)

_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_QR_RAW = "t=20260502T1234&s=500.00&fn=111&i=222&fp=333&n=1"


class _FakeResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._body


def _fake_async_client(body: object, calls: list[dict[str, Any]]) -> type:
    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append({"url": url, **kwargs})
            return _FakeResponse(body)

    return FakeAsyncClient


@pytest.mark.unit
async def test_fetch_receipt_posts_form_body(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    body = {
        "code": 1,
        "data": {
            "json": {
                "dateTime": "20260502T1234",
                "totalSum": 50000,
                "user": "Пятёрочка",
                "fiscalDriveNumber": "111",
                "fiscalDocumentNumber": "222",
                "fiscalSign": "333",
                "items": [],
            }
        },
    }
    monkeypatch.setattr(proverkacheka.httpx, "AsyncClient", _fake_async_client(body, calls))

    receipt = await ProverkaCheckaClient("secret-token").fetch_receipt(
        qr_raw=_QR_RAW,
        family_id=str(_FAMILY_ID),
        member_id=str(_MEMBER_ID),
    )

    assert receipt.total_amount == 500
    assert calls == [
        {
            "url": "https://proverkacheka.com/api/v1/check/get",
            "data": {"token": "secret-token", "qrraw": _QR_RAW},
        }
    ]
    assert "json" not in calls[0]


@pytest.mark.unit
async def test_fetch_receipt_code_5_has_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    body = {"code": 5, "data": "Нет информации по чеку (прочее)."}
    monkeypatch.setattr(proverkacheka.httpx, "AsyncClient", _fake_async_client(body, calls))

    with pytest.raises(ProverkaCheckaError) as exc_info:
        await ProverkaCheckaClient("secret-token").fetch_receipt(
            qr_raw=_QR_RAW,
            family_id=str(_FAMILY_ID),
            member_id=str(_MEMBER_ID),
        )

    message = str(exc_info.value)
    assert "Нет информации по чеку" in message
    assert "Unexpected" not in message


@pytest.mark.unit
async def test_fetch_receipt_code_401_points_to_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    body = {"code": 401, "data": "Не авторизован"}
    monkeypatch.setattr(proverkacheka.httpx, "AsyncClient", _fake_async_client(body, calls))

    with pytest.raises(ProverkaCheckaError) as exc_info:
        await ProverkaCheckaClient("secret-token").fetch_receipt(
            qr_raw=_QR_RAW,
            family_id=str(_FAMILY_ID),
            member_id=str(_MEMBER_ID),
        )

    assert "PROVERKACHEKA_API_TOKEN" in str(exc_info.value)
