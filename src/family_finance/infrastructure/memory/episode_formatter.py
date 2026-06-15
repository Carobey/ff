"""Converts domain Transaction objects into Graphiti episode text.

Format conventions (см. finance-domain skill):
  "<ИмяВладельца> купил(а) <описание> в <магазин> <дата> за <сумма> ₽"
  "Перевод <сумма> ₽ в <получатель> <дата>"

These short sentences are what Graphiti's entity extractor reads to build the
knowledge graph. The richer the description, the better the graph.

group_id convention: str(family_id) — all family members share one graph.
episode name: "{family_id}:{import_hash[:8]}" — globally unique, stable.
"""

from __future__ import annotations

import hashlib
import zoneinfo
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from family_finance.domain import Category, Direction, Transaction
from family_finance.domain.receipt import Receipt

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")

# Russian month names in genitive case (for "29 мая 2026")
_MONTHS_GEN = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


def _fmt_date(dt: datetime) -> str:
    """Format UTC datetime as '29 мая 2026' in Moscow time."""
    local = dt.astimezone(_MOSCOW)
    return f"{local.day} {_MONTHS_GEN[local.month]} {local.year}"


def _fmt_amount(tx: Transaction) -> str:
    """Format amount as '1 234 ₽'."""
    # Use space as thousands separator
    amount_int = int(tx.amount)
    frac = int((tx.amount - amount_int) * 100)
    formatted = f"{amount_int:,}".replace(",", " ")  # non-breaking space
    if frac:
        return f"{formatted},{frac:02d} ₽"
    return f"{formatted} ₽"


def transaction_to_episode_body(tx: Transaction, *, owner_name: str = "Юри") -> str:
    """Turn a Transaction into a short Russian sentence for Graphiti.

    Examples:
      "Юри купил продукты в Пятёрочка 4587 МСК 29 мая 2026 за 523 ₽"
      "Юри потратил 1 234 ₽ в WILDBERRIES 12 апреля 2026 (одежда)"
      "Перевод 5 000 ₽ — 3 июня 2026"
      "Доход 45 000 ₽ от Работодатель ООО 1 мая 2026"
    """
    date_str = _fmt_date(tx.occurred_at)
    amount_str = _fmt_amount(tx)
    merchant = tx.merchant_raw.strip()
    category_hint = tx.category.value if tx.category else ""

    if tx.direction == Direction.TRANSFER:
        return f"Перевод {amount_str} — {date_str}"

    if tx.direction == Direction.INCOME:
        source = f" от {merchant}" if merchant else ""
        return f"Доход {amount_str}{source} {date_str}"

    # EXPENSE
    cat_part = f" ({category_hint})" if category_hint else ""
    if merchant:
        return f"{owner_name} купил(а) в {merchant} {date_str} за {amount_str}{cat_part}"
    return f"{owner_name} потратил(а) {amount_str} {date_str}{cat_part}"


def make_episode_name(tx: Transaction) -> str:
    """Stable, unique episode name based on import_hash."""
    suffix = tx.import_hash[:12] if tx.import_hash else str(tx.transaction_id)[:12]
    return f"tx:{suffix}"


def make_receipt_episode_name(receipt: Receipt) -> str:
    """Stable, unique episode name for a whole receipt."""
    suffix = str(receipt.receipt_id)[:12]
    return f"receipt:{suffix}"


def receipt_to_episode_body(
    receipt: Receipt,
    transactions: Sequence[Transaction],
    *,
    owner_name: str = "Юри",
) -> str:
    """Format a whole receipt as a single episode body.

    One episode per receipt instead of per item keeps Graphiti's extraction
    cost bounded (~4 LLM calls per receipt, not 4×N items). The body still
    enumerates every item so the entity extractor has rich text to chew on.
    """
    date_str = _fmt_date(receipt.purchase_time)
    total_str = _fmt_amount_decimal(receipt.total_amount)
    store = (receipt.store_name or "магазине").strip()
    header = f"{owner_name} купил(а) в {store} {date_str} на сумму {total_str}"

    if not transactions:
        return header + "."

    item_lines = [
        f"- {tx.merchant_raw.split(' / ', 1)[-1]} "
        f"({tx.category.value}) — {_fmt_amount_decimal(tx.amount)}"
        for tx in transactions
    ]
    return header + ":\n" + "\n".join(item_lines)


def _fmt_amount_decimal(value: Decimal) -> str:
    """Format Decimal as '1 234,56 ₽' for episode text."""
    amount_int = int(value)
    frac = int((value - amount_int) * 100)
    formatted = f"{amount_int:,}".replace(",", " ")
    if frac:
        return f"{formatted},{frac:02d} ₽"
    return f"{formatted} ₽"


# ── Import summary episode (bulk CSV/PDF) ─────────────────────────────────────

# How many top categories to enumerate in the summary body. Bounds the episode
# text so Graphiti's extraction stays cheap on large imports.
_IMPORT_TOP_CATEGORIES = 8


def make_import_episode_name(transactions: Sequence[Transaction]) -> str:
    """Stable, unique episode name for one import batch (hash of its rows)."""
    digest = hashlib.sha256()
    for tx in sorted(transactions, key=lambda t: t.import_hash or str(t.transaction_id)):
        digest.update((tx.import_hash or str(tx.transaction_id)).encode())
    return f"import:{digest.hexdigest()[:12]}"


def import_to_episode_body(
    transactions: Sequence[Transaction],
    *,
    owner_name: str = "Юри",
) -> str:
    """Summarise a whole bulk import as ONE episode body (not per transaction).

    Per-tx episodes would cost ~4 LLM calls each (~400 for a 100-row CSV), so
    bulk imports write a single aggregate episode instead: period, total, and the
    largest expense categories. That still gives the entity extractor enough text
    to relate categories/periods, which is what CoachAgent retrieves later.
    """
    expenses = [tx for tx in transactions if tx.direction == Direction.EXPENSE]
    if not expenses:
        return f"{owner_name}: импорт выписки без расходных операций."

    occurred = [tx.occurred_at for tx in expenses]
    period = _fmt_period(min(occurred), max(occurred))
    total = sum((tx.amount for tx in expenses), Decimal("0"))

    by_category: dict[Category, tuple[Decimal, int]] = {}
    for tx in expenses:
        amount, count = by_category.get(tx.category, (Decimal("0"), 0))
        by_category[tx.category] = (amount + tx.amount, count + 1)

    ranked = sorted(by_category.items(), key=lambda kv: kv[1][0], reverse=True)
    header = (
        f"{owner_name}: импорт выписки {period} — {len(expenses)} операций "
        f"на {_fmt_amount_decimal(total)}"
    )
    lines = [
        f"- {category.value}: {_fmt_amount_decimal(amount)} ({count})"
        for category, (amount, count) in ranked[:_IMPORT_TOP_CATEGORIES]
    ]
    return header + ":\n" + "\n".join(lines)


def _fmt_period(start: datetime, end: datetime) -> str:
    """Format an import date range as '1–31 мая 2026' (Moscow time)."""
    s = start.astimezone(_MOSCOW)
    e = end.astimezone(_MOSCOW)
    if s.year == e.year and s.month == e.month:
        if s.day == e.day:
            return _fmt_date(start)
        return f"{s.day}–{e.day} {_MONTHS_GEN[e.month]} {e.year}"
    return f"{_fmt_date(start)} — {_fmt_date(end)}"
