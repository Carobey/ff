"""Ingest agent: bank statement file -> transactions -> Postgres."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

import structlog
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from family_finance.agents.state import FinanceState
from family_finance.application.ports import BankStatementParser
from family_finance.domain import Transaction
from family_finance.infrastructure.parsers import SberPdfParser, TinkoffCsvParser
from family_finance.infrastructure.persistence import PostgresTransactionRepository

logger = structlog.get_logger()


async def ingest_node(state: FinanceState) -> dict[str, object]:
    """Parse pending bank statement file (CSV or PDF) and persist transactions."""
    pending_csv = state.get("pending_csv")
    pending_pdf = state.get("pending_pdf")

    pending_path = pending_csv or pending_pdf
    if not pending_path:
        return {"messages": [AIMessage(content="Файл выписки не найден.")], "ingest_ok": False}

    family_id = uuid.UUID(state["family_id"])
    member_id = uuid.UUID(state["member_id"])
    path = Path(pending_path)
    content = await asyncio.to_thread(path.read_bytes)

    parser: BankStatementParser
    if pending_pdf:
        parser = SberPdfParser()
        bank_label = "Сбербанк PDF"
    else:
        parser = TinkoffCsvParser()
        bank_label = "Тинькофф CSV"

    try:
        transactions = await asyncio.to_thread(
            parser.parse,
            content,
            family_id=family_id,
            member_id=member_id,
            source_file=path.name,
        )
    except Exception:
        # Log full detail server-side; show the user a generic message so parser
        # internals (file paths, stack detail) don't leak into the chat.
        logger.exception("ingest_parse_failed", bank=bank_label, file=path.name)
        # Clear pending keys so stale state doesn't re-trigger ingest on next message
        return {
            "messages": [
                AIMessage(
                    content="❌ Не удалось разобрать файл выписки. Проверь формат и попробуй снова."
                )
            ],
            "pending_csv": None,
            "pending_pdf": None,
            "ingest_ok": False,
        }
    if not transactions:
        return {
            "messages": [
                AIMessage(content=f"В выписке ({bank_label}) не нашёл ни одной операции.")
            ],
            "pending_csv": None,
            "pending_pdf": None,
            "ingest_ok": False,
        }

    # HITL-пауза перед массовой записью в БД (ADR 0009). Всё ВЫШЕ — чистый парсинг
    # без сайд-эффектов, поэтому при resume нода переисполняется идемпотентно.
    # interrupt() требует checkpointer (PostgresSaver) — он всегда есть в рантайме.
    decision = interrupt(_import_preview(transactions, bank_label))
    if not _is_confirmed(decision):
        return {
            "messages": [AIMessage(content=f"🛑 Импорт выписки ({bank_label}) отменён.")],
            "pending_csv": None,
            "pending_pdf": None,
            "ingest_ok": False,
        }

    inserted = await PostgresTransactionRepository().add_many(transactions)
    message = (
        f"Импортировал {len(inserted)} новых транзакций из {len(transactions)} "
        f"строк выписки ({bank_label}). Дубли пропущены. Категоризирую…"
    )

    # NOTE: per-transaction Graphiti episodes are intentionally NOT written here.
    # Bulk CSV/PDF imports contain 100+ transactions → each add_episode costs
    # ~4 LLM calls, so 100 tx ≈ 400 LLM calls. Instead the categorizer writes ONE
    # aggregate episode per import (after categories are assigned), keeping cost
    # bounded while still feeding CoachAgent bulk-import context.
    #
    # We forward ONLY inserted rows to the categorizer so it does not re-classify
    # rows that already live in Postgres from a previous import.

    return {
        "messages": [AIMessage(content=message)],
        "parsed_transactions": inserted,
        "pending_csv": None,
        "pending_pdf": None,
        "current_intent": "upload_csv",
        "ingest_ok": bool(inserted),
    }


def _import_preview(transactions: list[Transaction], bank_label: str) -> dict[str, object]:
    """Deterministic confirmation payload surfaced to the user via ``interrupt()``.

    Числа считаются в Python (не LLM), даты — по ``occurred_at``. Бот форматирует
    этот dict в карточку с кнопками «Импортировать / Отмена».
    """
    occurred = [t.occurred_at for t in transactions]
    start: datetime = min(occurred)
    end: datetime = max(occurred)
    return {
        "kind": "import_confirm",
        "bank": bank_label,
        "count": len(transactions),
        "period_start": start.strftime("%d.%m.%Y"),
        "period_end": end.strftime("%d.%m.%Y"),
    }


def _is_confirmed(decision: object) -> bool:
    """Interpret the ``Command(resume=...)`` value as a yes/no decision.

    Кнопка присылает ``bool``; текстовый resume («да»/«нет») — строку. Любое
    не-утвердительное значение трактуется как отказ (fail-safe: не пишем в БД).
    """
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        return decision.strip().lower() in {"да", "yes", "y", "ок", "ok", "подтверждаю", "true"}
    return False
