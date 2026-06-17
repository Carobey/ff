"""
Supervisor — оркестратор графа.

На входе графа ``compact`` сворачивает длинный тред, затем supervisor извлекает
intent и pending-данные и маршрутизирует в Python (не в prompt): один интент →
одна specialist-нода через ``add_conditional_edges``; мульти-интент → веер ``Send``
на ``section_worker`` с join в ``synthesizer`` (orchestrator-worker, ADR 0007/0008).
Перед роутингом — injection guard. Все LLM-вызовы идут через маскирующий фасад.

Specialist-ноды: ingest, categorizer, receipt, ledger, coach, subscriptions,
budgets, advisor, tax, clarify, digest.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from family_finance.agents._messages import message_text
from family_finance.agents.advisor import advisor_node, build_advice_section, is_advice_question
from family_finance.agents.budgets import budgets_node, build_budgets_section, is_budgets_question
from family_finance.agents.categorizer import categorizer_node
from family_finance.agents.clarify import clarify_node, has_clarification_answers
from family_finance.agents.coach import coach_node
from family_finance.agents.compaction import compact_node
from family_finance.agents.ingest import ingest_node
from family_finance.agents.ledger import (
    build_spending_section,
    is_ledger_question,
    ledger_node,
    parse_period,
)
from family_finance.agents.receipt import receipt_node
from family_finance.agents.state import FinanceState, SectionResult
from family_finance.agents.subscriptions import (
    build_subscriptions_section,
    is_subscriptions_question,
    subscriptions_node,
)
from family_finance.agents.tax import build_tax_section, is_tax_question, tax_node
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.observability.langfuse_setup import emit_score
from family_finance.infrastructure.security import REFUSAL_MESSAGE, check_injection

logger = logging.getLogger(__name__)


_PATTERN_TOKENS = (
    "когда я",
    "когда последний",
    "последний раз",
    "как часто",
    "обычно",
    "аномалия",
    "паттерн",
    "привычк",
    "тенденц",
    "похож",
    "так тратил",
    "так же тратил",
    "в какой месяц",
    "самый дорогой месяц",
    "самый дешёвый",
)


def is_pattern_question(text: str) -> bool:
    """Detect behavioural finance questions for CoachAgent."""
    normalized = text.lower()
    return any(token in normalized for token in _PATTERN_TOKENS)


SUPERVISOR_SYSTEM = """Ты — финансовый помощник семьи Юри.

Помогаешь анализировать расходы, отвечаешь на вопросы о тратах по истории транзакций.
Отвечай кратко и дружелюбно (1-3 предложения), по-русски.

Если не понимаешь запрос — попроси уточнить.
"""


# ── Hybrid routing: LLM-планировщик (fallback по неоднозначности) ────────────
# Keyword fast-path остаётся первичным (дёшево, без LLM). Когда ни одно правило
# не сработало, LLM-планировщик выбирает НАБОР разделов: один (обычный ответ),
# несколько (мульти-интент без ключевых слов) или ничего (small-talk). Числа/SQL
# он не трогает — только маршрут (ADR 0007 + 0008).

_PlanSection = Literal["spending", "budgets", "subscriptions", "advice", "tax"]

_SECTION_TO_INTENT: dict[str, str] = {
    "spending": "query",
    "budgets": "budgets",
    "subscriptions": "subscriptions",
    "advice": "advice",
    "tax": "tax",
}


class QueryPlan(BaseModel):
    """Какие разделы финансов нужны для ответа (выбирает LLM по свободному тексту)."""

    model_config = ConfigDict(extra="forbid")

    sections: list[_PlanSection] = Field(
        default_factory=list,
        description=(
            "Нужные разделы: spending — суммы/траты по истории; budgets — бюджеты "
            "и лимиты; subscriptions — подписки/регулярные платежи; advice — совет "
            "как копить/экономить; tax — возврат НДФЛ по социальным вычетам "
            "(лечение/обучение/спорт/благотворительность). Несколько разделов → "
            "составной ответ. Пусто, если вопрос не про эти разделы."
        ),
    )
    pattern: bool = Field(
        default=False,
        description=(
            "true, если это вопрос про поведение/привычки «когда я последний раз "
            "так тратил» (тогда sections оставь пустым)."
        ),
    )


_PLAN_SYSTEM = """\
Ты — планировщик финансового помощника. По запросу пользователя определи, какие
разделы нужны для ответа. Не отвечай на сам вопрос и не считай числа.
- Один раздел → обычный ответ; несколько → составной ответ.
- Вопрос про привычки/поведение («когда я последний раз так тратил») —
  поставь pattern=true, sections оставь пустым.
- Приветствие/болтовня не про финансы — пустой sections и pattern=false.
"""


async def _plan_query(user_text: str) -> QueryPlan:
    """LLM picks the set of needed sections. Any failure → empty plan (smalltalk)."""
    try:
        model = cast(
            "Runnable[LanguageModelInput, QueryPlan]",
            get_chat_model(tier="worker").with_structured_output(QueryPlan),
        )
        return await model.ainvoke(
            [SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=user_text)]
        )
    except Exception:
        logger.exception("supervisor_plan_failed")
        return QueryPlan()


# ── Orchestrator-worker: мульти-интент (ADR 0008) ────────────────────────────
# Реальные запросы несут несколько интентов сразу («проанализируй расходы,
# проверь бюджеты, выяви подписки, дай рекомендации»). Одна ветка их теряет.
# Детектируем мульти ДЕТЕРМИНИРОВАННО: ≥2 разных интент-семейства по ключевым
# словам И явный разделитель клауз (запятая / точка с запятой). Это отделяет
# настоящие мульти-запросы от «сколько на подписки» (одна клауза → одиночный
# ledger). LLM в горячем пути не участвует — контракт ADR 0001/0007 сохранён.

_SECTION_ORDER: dict[str, int] = {
    "spending": 1,
    "budgets": 2,
    "subscriptions": 3,
    "advice": 4,
    "tax": 5,
}

_CLAUSE_SEPARATORS = (",", ";")


class SectionTask(TypedDict):
    """Полезная нагрузка одного ``Send`` к воркеру (вход узла ``section_worker``)."""

    kind: str
    family_id: str
    period_start: str | None
    period_end: str | None
    period_label: str


def _match_sections(text: str) -> list[str]:
    """Какие интент-семейства упомянуты в запросе (по ключевым словам)."""
    sections: list[str] = []
    if is_ledger_question(text):
        sections.append("spending")
    if is_budgets_question(text):
        sections.append("budgets")
    if is_subscriptions_question(text):
        sections.append("subscriptions")
    if is_advice_question(text):
        sections.append("advice")
    if is_tax_question(text):
        sections.append("tax")
    return sections


def _ordered_sections(sections: Sequence[str]) -> list[str]:
    return sorted(set(sections), key=lambda s: _SECTION_ORDER[s])


def _is_multi_intent(text: str) -> list[str]:
    """Return the ordered section plan if this is a multi-intent query, else []."""
    sections = _ordered_sections(_match_sections(text))
    if len(sections) >= 2 and any(sep in text for sep in _CLAUSE_SEPARATORS):
        return sections
    return []


def _multi_result(user_text: str, sections: list[str]) -> dict[str, object]:
    """Build the supervisor update that launches the worker веер.

    Период парсится ДЕТЕРМИНИРОВАННО один раз (общий для всех секций). Маркер
    ``current_intent="multi"`` свежий каждый ход — гейтит веер в route, не давая
    сработать устаревшему ``plan`` с прошлого хода; ``section_results=[]``
    сбрасывает накопитель под новую партию секций. Общий и для keyword-веера,
    и для LLM-планировщика (ADR 0008).
    """
    start, end, period_label = parse_period(user_text)
    return {
        "current_intent": "multi",
        "plan": sections,
        "period_start": start.isoformat() if start else None,
        "period_end": end.isoformat() if end else None,
        "period_label": period_label,
        "section_results": [],
    }


def _section_task(state: FinanceState, kind: str) -> SectionTask:
    return {
        "kind": kind,
        "family_id": state["family_id"],
        "period_start": state.get("period_start"),
        "period_end": state.get("period_end"),
        "period_label": state.get("period_label") or "за все время",
    }


async def section_worker(task: SectionTask) -> dict[str, object]:
    """Run one read-only section builder; result fans in via ``accumulate_sections``."""
    kind = task["kind"]
    family_id = uuid.UUID(task["family_id"])
    period_start = task.get("period_start")
    period_end = task.get("period_end")
    start = datetime.fromisoformat(period_start) if period_start else None
    end = datetime.fromisoformat(period_end) if period_end else None
    period_label = task.get("period_label") or "за все время"

    section: SectionResult
    if kind == "spending":
        section = await build_spending_section(
            family_id, start=start, end=end, period_label=period_label
        )
    elif kind == "budgets":
        section = await build_budgets_section(family_id, start=start, end=end)
    elif kind == "subscriptions":
        section = await build_subscriptions_section(family_id)
    elif kind == "advice":
        section = await build_advice_section(family_id, start=start, end=end)
    elif kind == "tax":
        section = await build_tax_section(family_id, start=start, end=end)
    else:
        logger.warning("section_worker_unknown_kind kind=%s", kind)
        return {}
    return {"section_results": [section]}


async def synthesizer_node(state: FinanceState) -> dict[str, object]:
    """Fan-in: stitch the computed sections into one coherent reply (ADR 0008).

    Числа уже посчитаны в Python внутри секций — синтез лишь упорядочивает их по
    ``order`` и склеивает, поэтому суммы не могут «уехать». Порядок прихода
    ``Send`` не гарантирован → сортируем явно.
    """
    sections = sorted(state.get("section_results", []), key=lambda s: s["order"])
    if not sections:
        return {
            "messages": [
                AIMessage(content="Не удалось собрать ответ — попробуй переформулировать.")
            ],
            "current_intent": "idle",
        }
    body = "\n\n".join(s["body"] for s in sections)
    return {"messages": [AIMessage(content=body)], "current_intent": "idle"}


async def supervisor_node(state: FinanceState) -> dict[str, object]:
    """Route to a specialist or answer directly via LLM.

    Routing-only branches do NOT emit a placeholder AIMessage: the specialist
    is responsible for the single user-visible reply. Otherwise the user would
    see two messages per request ("получил..." + actual result).
    """
    if state.get("pending_photo"):
        return {"current_intent": "upload_photo"}
    if state.get("pending_csv") or state.get("pending_pdf"):
        return {"current_intent": "upload_csv"}

    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = str(last_human.content) if last_human else "Привет"

    blocked = (await check_injection(user_text)).blocked
    emit_score("injection_blocked", 1.0 if blocked else 0.0)
    if blocked:
        return {
            "messages": [AIMessage(content=REFUSAL_MESSAGE)],
            "current_intent": "idle",
        }

    if has_clarification_answers(state, user_text):
        return {"current_intent": "clarify"}

    # Keyword-веер: ≥2 интент-семейства + разделитель клауз → мульти БЕЗ LLM.
    keyword_plan = _is_multi_intent(user_text)
    if keyword_plan:
        return _multi_result(user_text, keyword_plan)

    if is_tax_question(user_text):
        return {"current_intent": "tax"}
    # «Сколько откладывать / копить на цель» несёт слабый ledger-токен «сколько»,
    # но это совет, а не сводка трат. Перебиваем ledger advice'ом ТОЛЬКО когда нет
    # сильного спенд-слова (потрат/траты/расход) — иначе «сколько потратил и стоит
    # ли урезать траты» утекло бы в advice вместо сводки (QA-07).
    if is_ledger_question(user_text):
        strong_spend = any(w in user_text.lower() for w in ("потрат", "траты", "расход"))
        if strong_spend or not is_advice_question(user_text):
            return {"current_intent": "query"}
    if is_pattern_question(user_text):
        return {"current_intent": "pattern"}
    if is_subscriptions_question(user_text):
        return {"current_intent": "subscriptions"}
    if is_budgets_question(user_text):
        return {"current_intent": "budgets"}
    if is_advice_question(user_text):
        return {"current_intent": "advice"}

    # Keyword miss — LLM-планировщик выбирает НАБОР секций (ADR 0008): покрывает
    # бес-ключевые формулировки, в т.ч. мульти-интент без запятых. pattern →
    # coach; 0 секций → small-talk; 1 → одиночный маршрут; ≥2 → веер воркеров.
    plan = await _plan_query(user_text)
    if plan.pattern:
        return {"current_intent": "pattern"}
    sections = _ordered_sections(plan.sections)
    if len(sections) >= 2:
        return _multi_result(user_text, sections)
    if len(sections) == 1:
        return {"current_intent": _SECTION_TO_INTENT[sections[0]]}

    # Пустой план — small-talk: прямой ответ LLM (прежнее поведение).
    model = get_chat_model(tier="supervisor")
    response = await model.ainvoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM), HumanMessage(content=user_text)],
    )

    return {
        "messages": [AIMessage(content=message_text(response))],
        "current_intent": "idle",
    }


def route_after_supervisor(state: FinanceState) -> str | list[Send]:
    """Route to specialist nodes using explicit Python logic.

    Returns a node name (or the literal ``"end"`` sentinel mapped to ``END``
    by ``add_conditional_edges``), or a list of ``Send`` for the parallel
    multi-intent веер. ``current_intent == "multi"`` is set freshly by the
    planner each turn, so a stale ``plan`` from a previous turn can't fire.
    """
    if state.get("pending_photo"):
        return "receipt"
    if state.get("pending_csv") or state.get("pending_pdf"):
        return "ingest"
    if state.get("current_intent") == "multi":
        return [
            Send("section_worker", _section_task(state, kind)) for kind in state.get("plan", [])
        ]
    if state.get("current_intent") == "clarify":
        return "clarify"
    if state.get("current_intent") == "query":
        return "ledger"
    if state.get("current_intent") == "pattern":
        return "coach"
    if state.get("current_intent") == "subscriptions":
        return "subscriptions"
    if state.get("current_intent") == "budgets":
        return "budgets"
    if state.get("current_intent") == "advice":
        return "advisor"
    if state.get("current_intent") == "tax":
        return "tax"
    return "end"


def route_after_ingest(state: FinanceState) -> str:
    """Second branch point: categorize only when ingest produced new rows.

    A parser error or an all-duplicates import sets ``ingest_ok=False`` and
    short-circuits to END (ingest already emitted its own message), so the
    categorizer never runs on an empty batch.
    """
    return "categorizer" if state.get("ingest_ok") else "end"


def build_supervisor_graph(
    checkpointer: AsyncPostgresSaver | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Build LangGraph с persistence.

    В рантайме бот всегда передаёт реальный ``AsyncPostgresSaver``.
    ``checkpointer=None`` нужен только офлайн-утилитам (``just printgraph``):
    топология графа для отрисовки не зависит от persistence.
    """
    builder = StateGraph(FinanceState)
    builder.add_node("compact", compact_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("ingest", ingest_node)
    builder.add_node("categorizer", categorizer_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("ledger", ledger_node)
    builder.add_node("receipt", receipt_node)
    builder.add_node("coach", coach_node)
    builder.add_node("subscriptions", subscriptions_node)
    builder.add_node("budgets", budgets_node)
    builder.add_node("advisor", advisor_node)
    builder.add_node("tax", tax_node)
    # section_worker принимает узкую схему SectionTask (полезная нагрузка Send),
    # а не FinanceState — это допустимо для map-узлов LangGraph, но mypy этого не
    # выводит, поэтому приводим к Any точечно.
    builder.add_node("section_worker", cast("Any", section_worker))
    builder.add_node("synthesizer", synthesizer_node)
    # Компакция длинных тредов идёт ПЕРЕД supervisor: сворачивает старую историю
    # в сводку, не трогая маршрутизацию (ниже порога — no-op). ADR 0008.
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "ingest": "ingest",
            "clarify": "clarify",
            "ledger": "ledger",
            "receipt": "receipt",
            "coach": "coach",
            "subscriptions": "subscriptions",
            "budgets": "budgets",
            "advisor": "advisor",
            "tax": "tax",
            "section_worker": "section_worker",
            "end": END,
        },
    )
    # Веер воркеров сходится в синтез (join по суперстепу), синтез → END.
    builder.add_edge("section_worker", "synthesizer")
    builder.add_edge("synthesizer", END)
    # ingest → categorizer | END — 2nd branch point: skip categorizer on empty/failed import
    builder.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {"categorizer": "categorizer", "end": END},
    )
    builder.add_edge("categorizer", END)
    builder.add_edge("clarify", END)
    builder.add_edge("ledger", END)
    builder.add_edge("receipt", END)
    builder.add_edge("coach", END)
    builder.add_edge("subscriptions", END)
    builder.add_edge("budgets", END)
    builder.add_edge("advisor", END)
    builder.add_edge("tax", END)

    return builder.compile(checkpointer=checkpointer)
