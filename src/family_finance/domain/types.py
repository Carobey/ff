"""
Доменные типы: enum'ы и алиасы.

Принципы:
- Pure Python — никаких зависимостей кроме stdlib + pydantic
- StrEnum для совместимости с БД (TEXT-колонки) и JSON (сериализация)
- Категории — flat enum с dot-namespace для SQL LIKE-фильтрации
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

# Алиас для читаемости. ПРАВИЛО: всегда Decimal, никогда float — float теряет точность.
MoneyAmount = Decimal


def require_tz_aware(value: datetime) -> datetime:
    """Доменный инвариант: все datetime timezone-aware (внутри — UTC).

    Парсеры CSV/PDF/ОФД уже приводят к UTC; этот валидатор ловит naive-значения,
    просочившиеся мимо границы, вместо молчаливого принятия (см. CLAUDE.md «Время»).
    """
    if value.tzinfo is None:
        msg = "datetime must be timezone-aware (UTC inside), got naive"
        raise ValueError(msg)
    return value


class Currency(StrEnum):
    RUB = "RUB"
    USD = "USD"
    EUR = "EUR"


class Direction(StrEnum):
    EXPENSE = "expense"  # деньги ушли
    INCOME = "income"  # деньги пришли
    TRANSFER = "transfer"  # между своими кошельками — НЕ считать в расходах
    REFUND = "refund"  # возврат — корректирует прошлый expense


class TransactionSource(StrEnum):
    BANK_CSV = "bank_csv"
    BANK_PDF = "bank_pdf"  # Сбербанк — только PDF выписка
    RECEIPT_PHOTO_QR = "receipt_qr"  # ФНС вернула детализацию
    RECEIPT_PHOTO_OCR = "receipt_ocr"  # ФНС не нашла, голый OCR
    MANUAL = "manual"


class FamilyRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    GRANDPARENT = "grandparent"
    OTHER = "other"


class WalletPrivacy(StrEnum):
    SHARED = "shared"  # видят все взрослые в семье
    PRIVATE = "private"  # только владелец


class Category(StrEnum):
    """
    Flat enum с dot-namespace. Tax-агент Phase 3 фильтрует по префиксу:
    `cat.value.startswith("tax_ded.")`.
    """

    # Еда
    FOOD_GROCERIES = "food.groceries"
    FOOD_RESTAURANT = "food.restaurant"
    FOOD_DELIVERY = "food.delivery"

    # Транспорт
    TRANSPORT_FUEL = "transport.fuel"
    TRANSPORT_TAXI = "transport.taxi"
    TRANSPORT_PUBLIC = "transport.public"
    TRANSPORT_CARPARTS = "transport.carparts"

    # Дети
    KIDS_CLOTHES = "kids.clothes"
    KIDS_TOYS = "kids.toys"
    KIDS_SCHOOL = "kids.school"
    KIDS_ACTIVITIES = "kids.activities"

    # Покупки
    SHOPPING_CLOTHES = "shopping.clothes"
    SHOPPING_GENERIC = "shopping.generic"

    # Дом
    HOME_UTILITIES = "home.utilities"
    HOME_FURNITURE = "home.furniture"
    HOME_REPAIR = "home.repair"
    HOME_HOUSEHOLD = "home.household"

    # Здоровье (некомпенсируемое)
    HEALTH_PHARMACY = "health.pharmacy"
    HEALTH_GENERIC = "health.generic"

    # Развлечения
    ENTERTAINMENT_SUBS = "entertainment.subscriptions"
    ENTERTAINMENT_EVENTS = "entertainment.events"
    ENTERTAINMENT_HOBBIES = "entertainment.hobbies"

    PETS = "pets"

    # Налоговые вычеты — отдельный префикс для Phase 3 tax-агента (НК РФ)
    TAX_DED_MEDICAL = "tax_ded.medical"  # ст. 219 НК
    TAX_DED_EDUCATION = "tax_ded.education"  # ст. 219 НК
    TAX_DED_SPORT = "tax_ded.sport"  # ст. 219 НК (с 2022)
    TAX_DED_IIS = "tax_ded.iis"  # ст. 219.1 НК
    TAX_DED_PROPERTY = "tax_ded.property"  # ст. 220 НК

    # Доходы
    INCOME_SALARY = "income.salary"
    INCOME_OTHER = "income.other"

    # Спецслучаи
    TRANSFER_INTERNAL = "transfer.internal"  # между своими — НЕ расход
    UNCLASSIFIED = "unclassified"


class BankSource(StrEnum):
    """Поддерживаемые форматы банк-выписок."""

    TINKOFF = "tinkoff"
    SBER = "sber"
    ALFA = "alfa"
    GENERIC = "generic"
