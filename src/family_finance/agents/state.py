"""
LangGraph state. TypedDict + Annotated с reducer-ами (паттерн LangGraph 0.4+).

Принцип:
- Контейнер TypedDict (требование PostgresSaver)
- Значения полей — Pydantic-модели из domain (типизировано, валидируется)
- Reducer-ы для слияния параллельных апдейтов
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from family_finance.agents.clarifications import ClarificationQuestion
from family_finance.domain import Transaction

# === Reducers ===


def merge_transactions(
    existing: list[Transaction] | None,
    new: list[Transaction],
) -> list[Transaction]:
    """
    Сливаем по transaction_id: новая версия побеждает старую.

    Critical: categorizer может прислать апдейт ТОЙ ЖЕ транзакции
    с обновлённой категорией. operator.add (append) → дубли.
    """
    by_id: dict[str, Transaction] = {str(t.transaction_id): t for t in existing} if existing else {}
    for t in new:
        by_id[str(t.transaction_id)] = t
    return list(by_id.values())


def replace_open_questions(
    _existing: list[ClarificationQuestion] | None,
    new: list[ClarificationQuestion],
) -> list[ClarificationQuestion]:
    """Open clarification questions are replaced as one active batch."""
    return new


def accumulate_sections(
    existing: list[SectionResult] | None,
    new: list[SectionResult],
) -> list[SectionResult]:
    """Fan-in reducer for the orchestrator-worker веер (ADR 0008).

    Каждый воркер пишет одну секцию — они накапливаются в одном суперстепе
    (``operator.add``-семантика). Пустой апдейт ``[]`` от планировщика —
    сигнал «начать новую партию»: сбрасывает остаток прошлого мульти-запроса,
    иначе секции копились бы между ходами диалога (state checkpoint-ится).
    """
    if not new:
        return []
    return (existing or []) + list(new)


# === Intent (для routing supervisor'ом) ===

Intent = Literal[
    "upload_csv",  # пользователь прислал CSV-выписку
    "upload_photo",  # пользователь прислал фото чека
    "query",  # вопрос про данные ("сколько на еду в апреле")
    "pattern",  # вопрос про поведение ("когда я в последний раз так тратил")
    "subscriptions",  # запрос списка подписок / регулярных трат
    "budgets",  # запрос состояния бюджетов
    "advice",  # совет наставника: экономия / накопления (50/30/20, PYF)
    "tax",  # оценка возврата НДФЛ по социальным вычетам (ст. 219 НК)
    "multi",  # мульти-интентный запрос → веер воркеров + синтез (ADR 0008)
    "clarify",  # ответ на уточняющий вопрос бота
    "idle",  # нечего делать
]


class SectionResult(TypedDict):
    """Готовая секция ответа от одного воркера (orchestrator-worker, ADR 0008).

    ``body`` уже отрендерен детерминированно в Python (числа не от LLM);
    ``order`` задаёт порядок в синтезе (порядок прихода ``Send`` не гарантирован).
    """

    kind: str
    order: int
    title: str
    body: str


# === Главный state ===


class FinanceState(TypedDict, total=False):
    """
    State LangGraph. Передаётся между нодами, checkpoint-ится в PostgresSaver.

    total=False — не все поля обязаны быть установлены в каждом update.
    Ноды апдейтят только то что меняют.
    """

    # Диалог (стандартный reducer LangGraph)
    messages: Annotated[list[BaseMessage], add_messages]

    # Контекст пользователя (set один раз на старте)
    family_id: str
    member_id: str
    telegram_chat_id: int
    telegram_user_id: int

    # Pending I/O от Telegram-слоя
    pending_csv: str | None  # Тинькофф CSV
    pending_pdf: str | None  # Сбербанк PDF
    pending_photo: str | None
    pending_text: str | None

    # Результаты работы агентов
    parsed_transactions: Annotated[list[Transaction], merge_transactions]
    open_questions: Annotated[list[ClarificationQuestion], replace_open_questions]

    # Routing
    current_intent: Intent
    next_agent: str
    ingest_ok: bool  # ingest_node → gates the ingest→categorizer branch

    # Orchestrator-worker (мульти-интент, ADR 0008)
    plan: list[str]  # секции для веера: ["spending", "budgets", ...]
    period_start: str | None  # ISO; общий период, распарсенный планировщиком один раз
    period_end: str | None
    period_label: str
    section_results: Annotated[list[SectionResult], accumulate_sections]
