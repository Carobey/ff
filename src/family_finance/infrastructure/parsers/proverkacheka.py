"""ProverkaCheka (proverkacheka.com) API client.

Converts a fiscal QR string into a structured Receipt with ReceiptItem list.

API docs: https://proverkacheka.com/api
Registration: https://proverkacheka.com (free, ~2 min)
Add token to .env: PROVERKACHEKA_API_TOKEN=your-token-here

Flow:
  POST /api/v1/check/get
  Body: {"token": "<token>", "qrraw": "<qr_string>"}

  Response (success):
    {"code": 1, "data": {"json": { <fiscal detail> }}}

  Response (not found):
    {"code": 2, "data": "Данный чек не найден в ФНС"}
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import structlog

from family_finance.domain.receipt import Receipt, ReceiptItem

logger = structlog.get_logger()

_API_URL = "https://proverkacheka.com/api/v1/check/get"
_TIMEOUT = 15.0  # seconds


class ProverkaCheckaError(Exception):
    """Raised when the API returns an error or unexpected response."""


class ProverkaCheckaClient:
    """Async client for proverkacheka.com receipt detail API."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def fetch_receipt(
        self,
        *,
        qr_raw: str,
        family_id: str,
        member_id: str,
    ) -> Receipt:
        """Fetch fiscal receipt detail for *qr_raw* and return a Receipt domain object.

        Raises:
            ProverkaCheckaError: on API error, receipt not found, or parse failure.
        """
        payload = {"token": self._token, "qrraw": qr_raw}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(_API_URL, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ProverkaCheckaError(
                    f"HTTP {e.response.status_code} from ProverkaCheka"
                ) from e
            except httpx.RequestError as e:
                raise ProverkaCheckaError(f"Network error: {e}") from e

        body: dict[str, object] = resp.json()
        code = body.get("code")
        if code == 2:
            raise ProverkaCheckaError("Чек не найден в ФНС")
        if code != 1:
            raise ProverkaCheckaError(f"Unexpected code={code}: {body.get('data')}")

        raw_data = body.get("data", {})
        if not isinstance(raw_data, dict):
            raise ProverkaCheckaError("Unexpected data format from API")
        json_data = raw_data.get("json", {})
        if not isinstance(json_data, dict):
            raise ProverkaCheckaError("Missing json field in API response")

        return _build_receipt(json_data, qr_raw=qr_raw, family_id=family_id, member_id=member_id)


# ── Internal parsers ──────────────────────────────────────────────────────────


def _build_receipt(
    data: dict[str, object],
    *,
    qr_raw: str,
    family_id: str,
    member_id: str,
) -> Receipt:
    """Map ProverkaCheka JSON response to Receipt domain object."""
    # purchase_time: "20260502T1234" or "2026-05-02T12:34:00"
    purchase_time = _parse_datetime(str(data.get("dateTime", "")))

    # totalSum is in kopecks in the ФНС API. Some providers return strings
    # like "12345" (int-like) and others "12345.0" (float-like); Decimal(str)
    # tolerates both, plain int() would raise on the float form.
    total_amount = Decimal(str(data.get("totalSum", 0))) / Decimal("100")

    store_name = str(data.get("user", "") or data.get("retailPlace", "") or "").strip() or None

    items = _parse_items(data.get("items", []))

    return Receipt(
        family_id=uuid.UUID(family_id),
        member_id=uuid.UUID(member_id),
        qr_raw=qr_raw,
        fiscal_drive=str(data.get("fiscalDriveNumber", "") or ""),
        fiscal_document=str(data.get("fiscalDocumentNumber", "") or ""),
        fiscal_sign=str(data.get("fiscalSign", "") or ""),
        total_amount=total_amount,
        purchase_time=purchase_time,
        store_name=store_name,
        items=items,
        raw_response=data,
    )


def _parse_datetime(raw: str) -> datetime:
    """Parse ФНС datetime string.  Accepts 'YYYYMMDDTHHmm', 'YYYYMMDDTHHmmss',
    and ISO-like formats."""
    if not raw:
        return datetime.now(UTC)
    raw = raw.strip()
    # Try ISO first
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    # ФНС compact: 20260502T1234 or 20260502T123456. Пробуем без секунд ПЕРВЫМ:
    # strptime со «%H%M%S» рыхло матчит 4-значное время «1234» как 12:03:04,
    # поэтому короткий формат должен иметь приоритет (6-значное «123456» его не
    # пройдёт — останется «56» — и корректно уедет в формат с секундами).
    for fmt in ("%Y%m%dT%H%M", "%Y%m%dT%H%M%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    logger.warning("proverkacheka_datetime_unparsed", raw=raw, fallback="now(UTC)")
    return datetime.now(UTC)


def _parse_items(raw: object) -> list[ReceiptItem]:
    """Parse items array from ФНС JSON."""
    if not isinstance(raw, list):
        return []
    items: list[ReceiptItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            # ФНС quantities are integers (e.g. 1000 = 1.0)
            # but some providers return floats — Decimal(str(...)) is safe
            qty_raw = entry.get("quantity", 1)
            price_raw = entry.get("price", 0)
            total_raw = entry.get("sum", 0)
            nds_raw = entry.get("nds18Amount") or entry.get("ndsAmount")

            # Some APIs return kopecks, some return rubles — heuristic:
            # if total > 1_000_000 it's probably kopecks
            price = Decimal(str(price_raw))
            total = Decimal(str(total_raw))
            if total > Decimal("1000000"):
                price = price / Decimal("100")
                total = total / Decimal("100")

            items.append(
                ReceiptItem(
                    name=str(entry.get("name", "")).strip() or "?",
                    quantity=Decimal(str(qty_raw)),
                    price=price,
                    total=total,
                    nds_amount=Decimal(str(nds_raw)) if nds_raw is not None else None,
                )
            )
        except Exception:
            logger.warning("proverkacheka_item_unparsed", entry=entry)
    return items
