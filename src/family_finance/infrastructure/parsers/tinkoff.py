"""Tinkoff CSV statement parser."""

from __future__ import annotations

import csv
import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from io import StringIO
from zoneinfo import ZoneInfo

from family_finance.domain import (
    Category,
    Currency,
    Direction,
    Transaction,
    TransactionSource,
)

MOSCOW_TZ = ZoneInfo("Europe/Moscow")

TINKOFF_CATEGORY_MAP: dict[str, tuple[Direction, Category]] = {
    "Супермаркеты": (Direction.EXPENSE, Category.FOOD_GROCERIES),
    "Фастфуд": (Direction.EXPENSE, Category.FOOD_RESTAURANT),
    "Рестораны": (Direction.EXPENSE, Category.FOOD_RESTAURANT),
    "Аптеки": (Direction.EXPENSE, Category.HEALTH_PHARMACY),
    "Ремонт и мебель": (Direction.EXPENSE, Category.HOME_REPAIR),
    "Заправки": (Direction.EXPENSE, Category.TRANSPORT_FUEL),
    "Автоуслуги": (Direction.EXPENSE, Category.TRANSPORT_CARPARTS),
    "Мобильная связь": (Direction.EXPENSE, Category.HOME_UTILITIES),
    "Связь": (Direction.EXPENSE, Category.HOME_UTILITIES),
    "Книги и канцтовары": (Direction.EXPENSE, Category.KIDS_SCHOOL),
    "Переводы": (Direction.TRANSFER, Category.TRANSFER_INTERNAL),
}


class TinkoffCsvParser:
    """Parse current Tinkoff semicolon-separated CSV export."""

    def parse(
        self,
        content: bytes,
        *,
        family_id: uuid.UUID,
        member_id: uuid.UUID,
        source_file: str | None = None,
    ) -> list[Transaction]:
        text = self._decode(content)
        reader = csv.DictReader(StringIO(text), delimiter=";")
        self._validate_headers(reader.fieldnames)

        transactions: list[Transaction] = []
        for row in reader:
            if row.get("Статус") != "OK":
                continue
            transactions.append(
                self._parse_row(
                    row,
                    family_id=family_id,
                    member_id=member_id,
                    source_file=source_file,
                )
            )
        return transactions

    def _parse_row(
        self,
        row: dict[str, str],
        *,
        family_id: uuid.UUID,
        member_id: uuid.UUID,
        source_file: str | None,
    ) -> Transaction:
        payment_amount = self._parse_decimal(row["Сумма платежа"])
        bank_category = self._normalize_text(row["Категория"])
        direction, category = self._classify(bank_category, payment_amount)
        amount = abs(payment_amount)
        merchant_raw = row["Описание"].strip()
        occurred_at = self._parse_datetime(row["Дата операции"])
        posted_at = self._parse_date(row["Дата платежа"]) if row.get("Дата платежа") else None

        return Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=occurred_at,
            posted_at=posted_at,
            amount=amount,
            currency=Currency(row["Валюта платежа"]),
            direction=direction,
            merchant_raw=merchant_raw,
            category=category,
            confidence=self._confidence(direction, category),
            source=TransactionSource.BANK_CSV,
            source_file=source_file,
            import_hash=self._import_hash(occurred_at, amount, merchant_raw),
        )

    @staticmethod
    def _decode(content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8-sig")

    @staticmethod
    def _validate_headers(headers: Sequence[str] | None) -> None:
        required = {
            "Дата операции",
            "Статус",
            "Сумма платежа",
            "Валюта платежа",
            "Категория",
            "Описание",
        }
        missing = required - set(headers or [])
        if missing:
            raise ValueError(f"Tinkoff CSV missing columns: {', '.join(sorted(missing))}")

    @staticmethod
    def _parse_decimal(value: str) -> Decimal:
        normalized = value.strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
        return Decimal(normalized)

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.strptime(value.strip(), "%d.%m.%Y %H:%M:%S")
        return parsed.replace(tzinfo=MOSCOW_TZ).astimezone(UTC)

    @staticmethod
    def _parse_date(value: str) -> date:
        return datetime.strptime(value.strip(), "%d.%m.%Y").date()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.replace("\u00a0", " ").split())

    @staticmethod
    def _classify(bank_category: str, amount: Decimal) -> tuple[Direction, Category]:
        if bank_category in TINKOFF_CATEGORY_MAP:
            return TINKOFF_CATEGORY_MAP[bank_category]
        if amount > 0:
            return Direction.INCOME, Category.INCOME_OTHER
        return Direction.EXPENSE, Category.UNCLASSIFIED

    @staticmethod
    def _confidence(direction: Direction, category: Category) -> float:
        if category == Category.UNCLASSIFIED:
            return 0.0
        if direction == Direction.TRANSFER:
            return 0.6
        return 0.8

    @staticmethod
    def _import_hash(occurred_at: datetime, amount: Decimal, merchant_raw: str) -> str:
        payload = f"{occurred_at.isoformat()}|{amount}|{merchant_raw}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
