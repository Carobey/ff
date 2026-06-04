"""Ingest agent: bank statement file -> transactions -> Postgres."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage

from family_finance.agents.state import FinanceState
from family_finance.application.ports import BankStatementParser
from family_finance.infrastructure.parsers import SberPdfParser, TinkoffCsvParser
from family_finance.infrastructure.persistence import PostgresTransactionRepository


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
    except Exception as exc:
        # Clear pending keys so stale state doesn't re-trigger ingest on next message
        return {
            "messages": [AIMessage(content=f"❌ Не удалось разобрать файл: {exc}")],
            "pending_csv": None,
            "pending_pdf": None,
            "ingest_ok": False,
        }
    inserted = await PostgresTransactionRepository().add_many(transactions)
    message = (
        f"Импортировал {len(inserted)} новых транзакций из {len(transactions)} "
        f"строк выписки ({bank_label}). Дубли пропущены. Категоризирую…"
    )

    # NOTE: Graphiti episodic memory is intentionally NOT written here.
    # Bulk CSV/PDF imports contain 100+ transactions → each add_episode costs
    # ~4 LLM calls (entity + edge extraction + dedup), so 100 tx ≈ 400 LLM calls.
    # Graphiti is used only by ReceiptAgent for single real-time receipts.
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
