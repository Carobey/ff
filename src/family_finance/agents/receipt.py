"""ReceiptAgent: photo QR → ФНС API → Receipt → Transaction[].

Flow (P2-09..13):
  1. State carries pending_photo (path to downloaded image bytes)
  2. decode_qr() finds and decodes fiscal QR from the image
  3. ProverkaCheckaClient.fetch_receipt() returns Receipt + ReceiptItem[]
  4. _receipt_to_transactions() maps each item (or total) → Transaction
  5. Transactions are saved via PostgresTransactionRepository
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import structlog
from langchain_core.messages import AIMessage

from family_finance.agents.state import FinanceState
from family_finance.domain import Category, Currency, Direction, Transaction, TransactionSource
from family_finance.domain.receipt import Receipt
from family_finance.infrastructure.memory.episode_formatter import (
    make_receipt_episode_name,
    receipt_to_episode_body,
)
from family_finance.infrastructure.memory.graphiti_client import add_episode
from family_finance.infrastructure.parsers.proverkacheka import (
    ProverkaCheckaClient,
    ProverkaCheckaError,
)
from family_finance.infrastructure.parsers.qr_decoder import compose_fiscal_qr, decode_qr
from family_finance.infrastructure.parsers.vision_receipt import extract_fiscal_from_image
from family_finance.infrastructure.persistence import PostgresTransactionRepository
from family_finance.infrastructure.settings import get_settings

logger = structlog.get_logger()

# Keep strong references to background tasks so GC doesn't collect them prematurely
_bg_tasks: set[asyncio.Task[None]] = set()

# ── Simple keyword-based item categoriser ─────────────────────────────────────
# Good enough for Phase 2 demo; Phase 3 can replace with LLM structured output.

_ITEM_CATEGORY_RULES: list[tuple[tuple[str, ...], Category]] = [
    (
        (
            "молоко",
            "хлеб",
            "масло",
            "сыр",
            "яйц",
            "мясо",
            "рыб",
            "овощ",
            "фрукт",
            "кефир",
            "йогурт",
            "сметан",
            "творог",
            "колбас",
            "крупа",
            "мука",
            "сахар",
            "соль",
            "картоф",
            "помидор",
            "огурц",
        ),
        Category.FOOD_GROCERIES,
    ),
    (
        (
            "лекарств",
            "таблет",
            "витамин",
            "мазь",
            "капли",
            "сироп",
            "бинт",
            "шприц",
            "градусник",
            "тонометр",
        ),
        Category.HEALTH_PHARMACY,
    ),
    (
        ("подгузник", "пелёнк", "детск", "игрушк", "конструктор", "карандаш", "краск", "альбом"),
        Category.KIDS_TOYS,
    ),
    (("тетрадь", "учебник", "ранец", "школьн", "портфел"), Category.KIDS_SCHOOL),
    (("бензин", "дизель", "топлив"), Category.TRANSPORT_FUEL),
    (
        ("стиральн", "посудомоечн", "пылесос", "утюг", "чайник", "миксер", "блендер"),
        Category.HOME_HOUSEHOLD,
    ),
    (
        ("шуруп", "гвоздь", "краска", "обои", "ламинат", "плитк", "цемент", "кисточк", "ролик"),
        Category.HOME_REPAIR,
    ),
    (
        ("футболк", "брюки", "платье", "куртк", "пальто", "обувь", "кроссовк", "носки", "трусы"),
        Category.SHOPPING_CLOTHES,
    ),
]


def _categorise_item(name: str) -> Category:
    lower = name.lower()
    for tokens, cat in _ITEM_CATEGORY_RULES:
        if any(t in lower for t in tokens):
            return cat
    return Category.FOOD_GROCERIES  # safe default for shop receipts


# ── Receipt → Transaction mapper ──────────────────────────────────────────────


def _import_hash(occurred_at: datetime, amount: Decimal, name: str) -> str:
    payload = f"{occurred_at.isoformat()}|{amount}|{name}"
    return hashlib.sha256(payload.encode()).hexdigest()


def receipt_to_transactions(
    receipt: Receipt,
    *,
    family_id: uuid.UUID,
    member_id: uuid.UUID,
) -> list[Transaction]:
    """Convert Receipt items to Transaction list.

    One Transaction per ReceiptItem.  If items list is empty, create a single
    UNCLASSIFIED transaction for the receipt total.
    """
    transactions: list[Transaction] = []
    store = receipt.store_name or "чек"

    if receipt.items:
        for item in receipt.items:
            cat = item.predicted_category or _categorise_item(item.name)
            merchant = f"{store} / {item.name}"
            tx = Transaction(
                family_id=family_id,
                member_id=member_id,
                occurred_at=receipt.purchase_time,
                amount=item.total,
                currency=Currency.RUB,
                direction=Direction.EXPENSE,
                merchant_raw=merchant,
                category=cat,
                confidence=0.6,  # keyword match — moderate confidence
                source=TransactionSource.RECEIPT_PHOTO_QR,
                source_file=None,
                import_hash=_import_hash(receipt.purchase_time, item.total, merchant),
            )
            transactions.append(tx)
    else:
        # Fallback: one transaction for the whole receipt
        tx = Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=receipt.purchase_time,
            amount=receipt.total_amount,
            currency=Currency.RUB,
            direction=Direction.EXPENSE,
            merchant_raw=store,
            category=Category.FOOD_GROCERIES,
            confidence=0.4,
            source=TransactionSource.RECEIPT_PHOTO_QR,
            source_file=None,
            import_hash=_import_hash(receipt.purchase_time, receipt.total_amount, store),
        )
        transactions.append(tx)

    return transactions


# ── LangGraph node ────────────────────────────────────────────────────────────


async def receipt_node(state: FinanceState) -> dict[str, object]:
    """Process pending_photo: QR decode → ФНС → save transactions → narrative."""
    photo_path = state.get("pending_photo")
    if not photo_path:
        return {
            "messages": [AIMessage(content="Нет фото для обработки.")],
            "current_intent": "idle",
        }

    family_id = uuid.UUID(state["family_id"])
    member_id = uuid.UUID(state["member_id"])

    # 1. Read image bytes
    try:
        image_bytes = await _read_file(photo_path)
    except OSError as e:
        logger.error("receipt_node: cannot read photo", path=photo_path, error=str(e))
        return {
            "messages": [AIMessage(content="Не смог прочитать фото. Попробуй ещё раз.")],
            "current_intent": "idle",
            "pending_photo": None,
        }

    # 2. Decode QR  (pyzbar first, Vision LLM as fallback)
    qr_string = await asyncio.to_thread(decode_qr, image_bytes)

    if qr_string is None:
        # pyzbar failed — try Vision LLM to extract fiscal fields from receipt text
        logger.info("receipt_node: pyzbar failed, trying vision LLM fallback")
        fiscal_fields = await extract_fiscal_from_image(image_bytes)
        if fiscal_fields:
            qr_string = compose_fiscal_qr(fiscal_fields)
            logger.info(
                "receipt_node: vision fallback succeeded",
                fn=fiscal_fields.get("fn"),
                fd=fiscal_fields.get("i"),
            )
        else:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Не удалось считать QR и распознать данные чека. Попробуй:\n"
                            "• Сфотографировать ближе, чтобы QR-код занимал бо́льшую часть кадра\n"
                            "• Убедиться, что весь чек в кадре и хорошо освещён"
                        )
                    )
                ],
                "current_intent": "idle",
                "pending_photo": None,
            }

    # 3. Fetch receipt from ФНС via ProverkaCheka
    settings = get_settings()
    token = settings.proverkacheka_api_token
    if token is None:
        logger.warning("receipt_node: PROVERKACHEKA_API_TOKEN not set, skipping API call")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "QR считан, но токен ProverkaCheka не настроен. "
                        "Добавь PROVERKACHEKA_API_TOKEN в .env."
                    )
                )
            ],
            "current_intent": "idle",
            "pending_photo": None,
        }

    client = ProverkaCheckaClient(token.get_secret_value())
    try:
        receipt = await client.fetch_receipt(
            qr_raw=qr_string,
            family_id=str(family_id),
            member_id=str(member_id),
        )
    except ProverkaCheckaError as e:
        logger.warning("receipt_node: proverkacheka error", error=str(e))
        return {
            "messages": [AIMessage(content=f"ФНС вернула ошибку: {e}. Попробуй позже.")],
            "current_intent": "idle",
            "pending_photo": None,
        }

    # 4. Convert to transactions and save
    transactions = receipt_to_transactions(receipt, family_id=family_id, member_id=member_id)
    repo = PostgresTransactionRepository()
    inserted = await repo.add_many(transactions)

    # Fire-and-forget: Graphiti episodic memory (non-blocking).
    # ONE episode per receipt — not per item — so entity extraction costs
    # ~4 LLM calls instead of 4×N. The body lists every item, so the extractor
    # still sees rich text and produces useful edges.
    if inserted:
        group_id = str(family_id)
        task = asyncio.create_task(
            add_episode(
                name=make_receipt_episode_name(receipt),
                body=receipt_to_episode_body(receipt, inserted),
                source_description="receipt_qr",
                reference_time=receipt.purchase_time,
                group_id=group_id,
            )
        )
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    # 5. Build reply
    store = receipt.store_name or "магазин"
    total_fmt = f"{receipt.total_amount:,.2f} ₽".replace(",", " ")
    items_count = len(receipt.items)
    item_word = _items_word(items_count)

    reply = (
        f"✅ Чек от {store} на {total_fmt} обработан.\n"
        f"Позиций: {items_count} {item_word}, добавлено операций: {len(inserted)}."
    )
    if len(inserted) < len(transactions):
        reply += " (Часть дублей пропущена — уже были в базе.)"

    return {
        "messages": [AIMessage(content=reply)],
        "current_intent": "idle",
        "pending_photo": None,
    }


async def _read_file(path: str) -> bytes:
    return await asyncio.to_thread(Path(path).read_bytes)


def _items_word(n: int) -> str:
    if n % 100 in range(11, 20):
        return "позиций"
    r = n % 10
    if r == 1:
        return "позиция"
    if r in (2, 3, 4):
        return "позиции"
    return "позиций"
