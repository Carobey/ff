"""Sberbank PDF statement parser.

Сбербанк не предоставляет CSV — только PDF «Индивидуальная выписка
по счёту дебетовой карты». PDF не содержит табличных примитивов, поэтому
парсинг идёт по позиционированному тексту.

Формат строк в секции «Расшифровка операций»:

  Строка 1 (операция):
    <DD.MM.YYYY> <HH:MM>  <категория Сбера>  ...gap(5+ пробелов)...  <[+]сумма>
  Строка 2 (детали):
    <DD.MM.YYYY> <auth_code>  <описание/мерчант>. Операция по карте ****XXXX

Секция начинается после «и код авторизации» и заканчивается до
«Дата формирования документа».
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
from zoneinfo import ZoneInfo

import pdfplumber

from family_finance.domain import (
    Category,
    Currency,
    Direction,
    Transaction,
    TransactionSource,
)

MOSCOW_TZ = ZoneInfo("Europe/Moscow")

# ── Section boundaries ────────────────────────────────────────────────────────

_SECTION_START = "и код авторизации"
_SECTION_END = "Дата формирования документа"

# ── Line regexes ──────────────────────────────────────────────────────────────

# Строка 1: дата, время, категория, сумма (сумма справа, перед ней ≥5 пробелов)
_TX_LINE1 = re.compile(
    r"^\s*"
    r"(\d{2}\.\d{2}\.\d{4})"  # дата операции (DD.MM.YYYY)
    r"\s+"
    r"(\d{2}:\d{2})"  # время HH:MM
    r"\s+"
    r"(.+?)"  # категория Сбера (non-greedy)
    r"\s{5,}"  # правый отступ ≥5 пробелов
    r"(\+?\d[\d ]*,\d{2})"  # сумма: необязательный +, цифры, запятая, 2 знака
    r"\s*$"
)

# Строка 2: дата обработки, код авторизации, описание операции
_TX_LINE2 = re.compile(
    r"^\s*"
    r"\d{2}\.\d{2}\.\d{4}"  # дата обработки (не захватываем)
    r"\s+"
    r"\d+"  # код авторизации (не захватываем)
    r"\s+"
    r"(.+?)$"  # описание мерчанта
)

# Убрать суффикс «. Операция по карте ****XXXX» из описания
_CARD_SUFFIX_RE = re.compile(
    r"\.\s*Операция по карте\s+[•\*]+\d+\s*$",
    re.IGNORECASE,
)

# ── Категории Сбербанка ───────────────────────────────────────────────────────

SBER_CATEGORY_MAP: dict[str, tuple[Direction, Category]] = {
    "Супермаркеты": (Direction.EXPENSE, Category.FOOD_GROCERIES),
    "Продукты": (Direction.EXPENSE, Category.FOOD_GROCERIES),
    "Рестораны и кафе": (Direction.EXPENSE, Category.FOOD_RESTAURANT),
    "Кафе и рестораны": (Direction.EXPENSE, Category.FOOD_RESTAURANT),
    "Фастфуд": (Direction.EXPENSE, Category.FOOD_RESTAURANT),
    "Доставка еды": (Direction.EXPENSE, Category.FOOD_DELIVERY),
    "Транспорт": (Direction.EXPENSE, Category.TRANSPORT_PUBLIC),
    "Такси": (Direction.EXPENSE, Category.TRANSPORT_TAXI),
    "АЗС": (Direction.EXPENSE, Category.TRANSPORT_FUEL),
    "Заправки": (Direction.EXPENSE, Category.TRANSPORT_FUEL),
    "Одежда и обувь": (Direction.EXPENSE, Category.SHOPPING_CLOTHES),
    "Одежда": (Direction.EXPENSE, Category.SHOPPING_CLOTHES),
    "Здоровье": (Direction.EXPENSE, Category.HEALTH_GENERIC),
    "Аптека": (Direction.EXPENSE, Category.HEALTH_PHARMACY),
    "Красота": (Direction.EXPENSE, Category.HEALTH_GENERIC),
    "Образование": (Direction.EXPENSE, Category.KIDS_ACTIVITIES),
    "ЖКХ": (Direction.EXPENSE, Category.HOME_UTILITIES),
    "Коммунальные платежи": (Direction.EXPENSE, Category.HOME_UTILITIES),
    "Связь": (Direction.EXPENSE, Category.HOME_UTILITIES),
    "Отдых и развлечения": (Direction.EXPENSE, Category.ENTERTAINMENT_EVENTS),
    "Развлечения": (Direction.EXPENSE, Category.ENTERTAINMENT_EVENTS),
    "Кино": (Direction.EXPENSE, Category.ENTERTAINMENT_EVENTS),
    "Подписки": (Direction.EXPENSE, Category.ENTERTAINMENT_SUBS),
    "Зарплата": (Direction.INCOME, Category.INCOME_SALARY),
    "Прочие операции": (Direction.EXPENSE, Category.UNCLASSIFIED),
    "Другое": (Direction.EXPENSE, Category.UNCLASSIFIED),
}

# Категории переводов — direction определяется знаком суммы
_TRANSFER_CATEGORIES = frozenset(
    {
        "Переводы",
        "Перевод на карту",
        "Перевод с карты",
        "Перевод СБП",
        "Перевод между своими счетами",
        "Выдача наличных",
        "Пополнение карты",
    }
)


# ── Parser ────────────────────────────────────────────────────────────────────


class SberPdfParser:
    """Parse Sberbank debit card PDF statement via text extraction."""

    def parse(
        self,
        content: bytes,
        *,
        family_id: uuid.UUID,
        member_id: uuid.UUID,
        source_file: str | None = None,
    ) -> list[Transaction]:
        text = self._extract_text(content)
        raw_rows = self._parse_transactions(text)
        transactions: list[Transaction] = []
        for row in raw_rows:
            tx = self._build_transaction(
                row, family_id=family_id, member_id=member_id, source_file=source_file
            )
            if tx is not None:
                transactions.append(tx)
        return transactions

    # ── Text extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: bytes) -> str:
        """Concatenate layout-preserving text from all pages."""
        pages: list[str] = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text(layout=True) or "")
        return "\n".join(pages)

    # ── Transaction section parsing ───────────────────────────────────────────

    @staticmethod
    def _find_section(text: str) -> str:
        start = text.find(_SECTION_START)
        end = text.find(_SECTION_END)
        if start == -1:
            return ""
        section_start = start + len(_SECTION_START)
        section_end = end if end != -1 else len(text)
        return text[section_start:section_end]

    def _parse_transactions(self, text: str) -> list[dict[str, str]]:
        section = self._find_section(text)
        if not section:
            return []

        rows: list[dict[str, str]] = []
        pending: dict[str, str] | None = None

        for line in section.splitlines():
            m1 = _TX_LINE1.match(line)
            if m1:
                if pending:
                    # Line 2 was missing — keep what we have
                    rows.append(pending)
                pending = {
                    "date_str": m1.group(1),
                    "time_str": m1.group(2),
                    "sber_category": m1.group(3).strip(),
                    "amount_raw": m1.group(4).strip(),
                    "merchant": m1.group(3).strip(),  # fallback if no line 2
                }
                continue

            if pending is not None:
                m2 = _TX_LINE2.match(line)
                if m2:
                    raw = m2.group(1).strip()
                    merchant = _CARD_SUFFIX_RE.sub("", raw).strip()
                    pending["merchant"] = merchant or pending["sber_category"]
                    rows.append(pending)
                    pending = None

        if pending is not None:
            rows.append(pending)

        return rows

    # ── Transaction builder ───────────────────────────────────────────────────

    def _build_transaction(
        self,
        row: dict[str, str],
        *,
        family_id: uuid.UUID,
        member_id: uuid.UUID,
        source_file: str | None,
    ) -> Transaction | None:
        try:
            amount, is_positive = self._parse_amount(row["amount_raw"])
        except (InvalidOperation, ValueError):
            return None

        direction, category = self._classify(row["sber_category"], is_positive)
        occurred_at = self._parse_datetime(row["date_str"], row["time_str"])

        return Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=occurred_at,
            amount=amount,
            currency=Currency.RUB,
            direction=direction,
            merchant_raw=row["merchant"],
            category=category,
            confidence=self._confidence(category),
            source=TransactionSource.BANK_PDF,
            source_file=source_file,
            import_hash=self._import_hash(occurred_at, amount, row["merchant"]),
        )

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_amount(raw: str) -> tuple[Decimal, bool]:
        """Parse «[+]NN NNN,NN» → (Decimal, is_credit)."""
        is_positive = raw.lstrip().startswith("+")
        normalized = re.sub(r"\s", "", raw.lstrip("+")).replace(",", ".")
        return Decimal(normalized), is_positive

    @staticmethod
    def _classify(sber_category: str, is_positive: bool) -> tuple[Direction, Category]:
        if sber_category in _TRANSFER_CATEGORIES:
            if is_positive:
                return Direction.INCOME, Category.INCOME_OTHER
            return Direction.TRANSFER, Category.TRANSFER_INTERNAL

        if is_positive:
            return Direction.INCOME, Category.INCOME_OTHER

        return SBER_CATEGORY_MAP.get(sber_category, (Direction.EXPENSE, Category.UNCLASSIFIED))

    @staticmethod
    def _confidence(category: Category) -> float:
        if category == Category.UNCLASSIFIED:
            return 0.0
        if category in (Category.INCOME_OTHER, Category.TRANSFER_INTERNAL):
            return 0.6
        return 0.8

    @staticmethod
    def _parse_datetime(date_str: str, time_str: str) -> datetime:
        parsed = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
        return parsed.replace(tzinfo=MOSCOW_TZ).astimezone(UTC)

    @staticmethod
    def _import_hash(occurred_at: datetime, amount: Decimal, merchant: str) -> str:
        payload = f"{occurred_at.isoformat()}|{amount}|{merchant}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
