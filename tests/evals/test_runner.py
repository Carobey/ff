"""
Eval runner — pytest tests marked with @pytest.mark.eval.

Run with:  just eval   OR   uv run pytest -m eval -v
Each categorization test calls the real LLM (slow, costs money).
CSV-parsing tests are deterministic (no LLM).

LangFuse traces: appear in dashboard under tag "eval" when LANGFUSE_* env is set.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import asyncpg
import pytest
import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from family_finance.agents._messages import message_text
from family_finance.agents.categorizer import CategoryPrediction, build_system_prompt
from family_finance.agents.coach import coach_node
from family_finance.agents.supervisor import route_after_supervisor, supervisor_node
from family_finance.domain import Category, Direction, Transaction, TransactionSource
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.parsers import TinkoffCsvParser
from family_finance.infrastructure.persistence import (
    PostgresCategoryCatalog,
    PostgresTransactionRepository,
)
from family_finance.infrastructure.security import check_injection
from family_finance.infrastructure.settings import get_settings
from tests.evals.cascade_task import run_cascade
from tests.evals.scorers import apply_scorer

# ── LLM-as-judge scorer ───────────────────────────────────────────────────────


class _CategoryJudge(BaseModel):
    """Verdict of the LLM-judge on whether a category fits a merchant."""

    model_config = ConfigDict(extra="forbid")

    reasonable: bool
    reason: str = ""


_JUDGE_SYSTEM = (
    "Ты — судья качества категоризации трат. Тебе дают название продавца и "
    "категорию, которую присвоил классификатор. Верни reasonable=true, если "
    "категория разумна для этого продавца, иначе reasonable=false."
)


async def _llm_judge_category(merchant_raw: str, predicted_category: str) -> float:
    """LLM-as-judge: 1.0 if the predicted category is reasonable for the merchant."""
    model = get_chat_model(tier="worker").with_structured_output(_CategoryJudge)
    verdict: _CategoryJudge = await model.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=f"Продавец: {merchant_raw}\nКатегория: {predicted_category}"),
        ],
    )
    return 1.0 if verdict.reasonable else 0.0


class _PlanJudge(BaseModel):
    """Verdict of the LLM-judge on whether a section plan fits a query."""

    model_config = ConfigDict(extra="forbid")

    reasonable: bool
    reason: str = ""


_PLAN_JUDGE_SYSTEM = (
    "Ты — судья качества планировщика финансового помощника. Тебе дают запрос "
    "пользователя и НАБОР разделов, который выбрал планировщик: spending (траты), "
    "budgets (бюджеты), subscriptions (подписки), advice (советы как экономить). "
    "Верни reasonable=true, если набор разделов разумно покрывает запрос (ничего "
    "важного не упущено и нет явно лишнего), иначе reasonable=false."
)


async def _llm_judge_plan(user_text: str, plan: list[str]) -> float:
    """LLM-as-judge: 1.0 if the chosen section set reasonably covers the query."""
    model = get_chat_model(tier="worker").with_structured_output(_PlanJudge)
    verdict: _PlanJudge = await model.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(content=_PLAN_JUDGE_SYSTEM),
            HumanMessage(content=f"Запрос: {user_text}\nРазделы: {plan}"),
        ],
    )
    return 1.0 if verdict.reasonable else 0.0


class _CoachJudge(BaseModel):
    """Verdict of the LLM-judge on whether a coach answer is grounded."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    reason: str = ""


_COACH_JUDGE_SYSTEM = (
    "Ты — судья качества финансового коуча. Тебе дают поведенческий вопрос "
    "пользователя о его тратах и ответ коуча. Верни grounded=true, если ответ по "
    "делу отвечает на вопрос конкретными цифрами/датами, а НЕ отговоркой «нет "
    "данных» и не общими словами; иначе grounded=false."
)


async def _llm_judge_coach(user_text: str, answer: str) -> float:
    """LLM-as-judge: 1.0 if the coach answer addresses the question with real data."""
    model = get_chat_model(tier="worker").with_structured_output(_CoachJudge)
    verdict: _CoachJudge = await model.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(content=_COACH_JUDGE_SYSTEM),
            HumanMessage(content=f"Вопрос: {user_text}\nОтвет коуча: {answer}"),
        ],
    )
    return 1.0 if verdict.grounded else 0.0


# ── Constants ─────────────────────────────────────────────────────────────────

EVAL_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
EVAL_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
CASES_DIR = Path(__file__).parent / "cases"

# Выделенная одноразовая семья для coach-evals (не пересекается с реальными chat-id).
_COACH_EVAL_TG_ID = 999_000_222
# Фиксированная майская история доставок: даты в прошлом и стабильны между
# прогонами, поэтому детерминированный import_hash держит add_many идемпотентным
# (count не плывёт). Последняя доставка 27.05.2026 → эта дата должна попасть в
# ответ «когда последний раз»; число майских доставок (3) → в ответ «как часто».
_COACH_DELIVERY_DATES = (date(2026, 5, 8), date(2026, 5, 16), date(2026, 5, 27))


def _load_cases(agent: str) -> list[Path]:
    return sorted((CASES_DIR / agent).glob("*.yaml"))


# Multi-веер кладёт явный ``plan``; одиночный маршрут планировщика — только
# ``current_intent``. Восстанавливаем секцию из интента, чтобы judge видел
# реальный выбор, а не пустой список (pattern/idle → не секция).
_INTENT_TO_SECTION: dict[str, str] = {
    "query": "spending",
    "budgets": "budgets",
    "subscriptions": "subscriptions",
    "advice": "advice",
}


def _extract_plan(update: dict[str, Any]) -> list[str]:
    """The section set the supervisor chose: explicit веер plan or a single route."""
    plan = update.get("plan")
    if plan:
        return list(plan)
    section = _INTENT_TO_SECTION.get(str(update.get("current_intent")))
    return [section] if section else []


async def _skip_if_postgres_unavailable() -> None:
    try:
        conn = await asyncpg.connect(dsn=get_settings().database_url.get_secret_value())
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres is unavailable: {exc}")
    else:
        await conn.close()


# ── Categorization evals (LLM) ────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("categorization"),
    ids=lambda p: p.stem,
)
async def test_categorization(case_path: Path) -> None:
    """Run a single transaction through the LLM and score the result."""
    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)
    inp = case["input"]
    tx = Transaction(
        family_id=EVAL_FAMILY_ID,
        member_id=EVAL_MEMBER_ID,
        occurred_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC),
        amount=Decimal(inp["amount"]),
        currency="RUB",  # type: ignore[arg-type]
        direction=Direction(inp["direction"]),
        merchant_raw=inp["merchant_raw"],
        category=Category.UNCLASSIFIED,
        confidence=0.0,
        source=TransactionSource.BANK_CSV,
        import_hash=f"eval-{case['id']}",
    )

    model = get_chat_model(tier="worker").with_structured_output(CategoryPrediction)
    # Note: LangFuse callbacks are omitted here — they conflict with pytest-asyncio's
    # per-test event loop lifecycle (OTLP exporter cleanup races the loop closure).
    # When running evals against a live LangFuse instance, use `just eval` from a
    # long-lived process (e.g. the bot) where the loop outlives the HTTP cleanup.
    system_prompt = build_system_prompt(await PostgresCategoryCatalog().render_taxonomy())
    prediction: CategoryPrediction = await model.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Продавец: {tx.merchant_raw}\nСумма: {tx.amount} ₽"),
        ],
    )

    result: dict[str, Any] = {
        "category": prediction.category.value,
        "confidence": prediction.confidence,
    }

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        if scorer_cfg["type"] == "llm_judge":
            score = await _llm_judge_category(inp["merchant_raw"], result["category"])
        else:
            score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            failures.append(
                f"scorer={scorer_cfg['type']} field={scorer_cfg.get('field')} "
                f"got={result} expected={case['expected']}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── Cascade evals (deterministic, live DB — no LLM) ──────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("cascade"),
    ids=lambda p: p.stem,
)
async def test_cascade(case_path: Path) -> None:
    """Run a merchant through the rule cascade against the live DB — no LLM.

    Covers the «узнать продавца без LLM» path and the learning loop (ответ
    юзера → правило семьи). Skips when Postgres is unavailable.
    """
    await _skip_if_postgres_unavailable()

    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)

    result = await run_cascade(case["input"])

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            field = scorer_cfg["field"]
            failures.append(
                f"scorer={scorer_cfg['type']} field={field} "
                f"got={result.get(field)} expected={case['expected'].get(field)}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── CSV-parsing evals (deterministic) ────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("csv_parsing"),
    ids=lambda p: p.stem,
)
def test_csv_parsing(case_path: Path) -> None:
    """Parse inline CSV content and score the result — no LLM needed."""
    case: dict[str, Any] = yaml.safe_load(case_path.read_text())

    csv_bytes = case["input"]["csv_content"].encode("utf-8")
    transactions = TinkoffCsvParser().parse(
        csv_bytes,
        family_id=EVAL_FAMILY_ID,
        member_id=EVAL_MEMBER_ID,
    )

    result: dict[str, Any] = {
        "count": len(transactions),
        "first_category": transactions[0].category.value if transactions else None,
        "has_transfer": any(tx.direction == Direction.TRANSFER for tx in transactions),
    }

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            field = scorer_cfg["field"]
            failures.append(
                f"scorer={scorer_cfg['type']} field={field} "
                f"got={result.get(field)} expected={case['expected'].get(field)}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── Security evals (prompt injection) ────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("security"),
    ids=lambda p: p.stem,
)
async def test_security_injection(case_path: Path) -> None:
    """Run a jailbreak attempt through the injection guard and score the verdict.

    Deterministic-pattern cases score without an LLM; the paraphrased case
    escalates to the keyword-gated LLM-judge.
    """
    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)

    verdict = await check_injection(case["input"]["text"])
    result: dict[str, Any] = {"blocked": verdict.blocked}

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            field = scorer_cfg["field"]
            failures.append(
                f"scorer={scorer_cfg['type']} field={field} "
                f"got={result.get(field)} expected={case['expected'].get(field)}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── Tool-call correctness evals (routing) ─────────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("tool_routing"),
    ids=lambda p: p.stem,
)
async def test_tool_routing(case_path: Path) -> None:
    """Tool-call correctness: supervisor must route to the expected specialist.

    Inputs are chosen to hit the deterministic intent classifiers, so the
    supervisor resolves the route without an LLM call.
    """
    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)

    state: dict[str, Any] = {"messages": [HumanMessage(content=case["input"]["text"])]}
    update = await supervisor_node(state)
    route = route_after_supervisor({**state, **update})
    result: dict[str, Any] = {"route": route}

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            field = scorer_cfg["field"]
            failures.append(
                f"scorer={scorer_cfg['type']} field={field} "
                f"got={result.get(field)} expected={case['expected'].get(field)}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── Multi-intent planner evals (orchestrator веер, ADR 0008) ─────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("multi_intent"),
    ids=lambda p: p.stem,
)
async def test_multi_intent(case_path: Path) -> None:
    """Planner correctness: supervisor must fan out to the expected section set.

    Keyword-based cases resolve the plan deterministically (no LLM); keyword-less
    cases escalate to the LLM-планировщик and are scored by an LLM-judge on
    whether the chosen section set reasonably covers the query (ADR 0008). We
    assert only the PLAN (which workers fire), not the synthesized answer — the
    eval harness has no live MCP/Postgres seed data.
    """
    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)

    state: dict[str, Any] = {"messages": [HumanMessage(content=case["input"]["text"])]}
    update = await supervisor_node(state)
    result: dict[str, Any] = {"plan": _extract_plan(update)}

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        if scorer_cfg["type"] == "llm_judge":
            score = await _llm_judge_plan(case["input"]["text"], result["plan"])
        else:
            score = apply_scorer(scorer_cfg, result, case["expected"])
        if score < 1.0:
            failures.append(
                f"scorer={scorer_cfg['type']} field={scorer_cfg.get('field')} "
                f"got={result} expected={case['expected']}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"


# ── Coach quality evals (LLM + MCP tools, live DB) ───────────────────────────


async def _seed_coach_history() -> uuid.UUID:
    """Seed a fixed delivery history for the coach-eval family; return family_id.

    Idempotent: deterministic import_hash + add_many ON CONFLICT DO NOTHING.
    """
    repo = PostgresTransactionRepository()
    family_id, member_id = await repo.ensure_member_for_telegram(
        telegram_user_id=_COACH_EVAL_TG_ID,
        name="Coach eval family",
    )
    txs = [
        Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC),
            amount=Decimal("1100.00"),
            currency="RUB",  # type: ignore[arg-type]
            direction=Direction.EXPENSE,
            merchant_raw="Яндекс Еда",
            merchant_normalized="Яндекс Еда",
            category=Category.FOOD_DELIVERY,
            confidence=0.95,
            source=TransactionSource.BANK_CSV,
            import_hash=f"coach-eval:delivery:{d.isoformat()}",
        )
        for d in _COACH_DELIVERY_DATES
    ]
    # Трата позже последней доставки — чтобы «последний раз ДОСТАВКУ» требовал
    # фильтра по продавцу, а не просто «последняя трата».
    txs.append(
        Transaction(
            family_id=family_id,
            member_id=member_id,
            occurred_at=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
            amount=Decimal("4200.00"),
            currency="RUB",  # type: ignore[arg-type]
            direction=Direction.EXPENSE,
            merchant_raw="Пятёрочка",
            merchant_normalized="Пятёрочка",
            category=Category.FOOD_GROCERIES,
            confidence=0.95,
            source=TransactionSource.BANK_CSV,
            import_hash="coach-eval:grocery:2026-05-30",
        )
    )
    await repo.add_many(txs)
    return family_id


def _coach_expected_tokens(kind: str) -> list[str]:
    """Acceptable substrings for the tool-derived number/date, by case kind."""
    if kind == "last_delivery_date":
        last = _COACH_DELIVERY_DATES[-1]
        return [f"{last.day:02d}.{last.month:02d}", f"{last.day} мая"]
    if kind == "delivery_count":
        return [str(len(_COACH_DELIVERY_DATES))]
    raise ValueError(f"Unknown coach expected kind: {kind!r}")


@pytest.mark.eval
@pytest.mark.parametrize(
    "case_path",
    _load_cases("coach"),
    ids=lambda p: p.stem,
)
async def test_coach_quality(case_path: Path) -> None:
    """Coach answer QUALITY: a behavioural question must be answered with a
    number/date that came from the MCP tools (grounding), not hallucinated.

    Seeds a fixed delivery history for a throwaway family, runs the real ReAct
    coach (LLM + MCP tools, live DB) and asserts the answer echoes the
    tool-derived token. Skips when Postgres is unavailable.
    """
    await _skip_if_postgres_unavailable()

    text = await asyncio.to_thread(case_path.read_text)
    case: dict[str, Any] = yaml.safe_load(text)

    family_id = await _seed_coach_history()
    state: dict[str, Any] = {
        "family_id": str(family_id),
        "messages": [HumanMessage(content=case["input"]["text"])],
    }
    update = await coach_node(state)
    answer = message_text(cast("BaseMessage", update["messages"][-1]))  # type: ignore[index]
    result: dict[str, Any] = {"answer": answer}

    expected = dict(case["expected"])
    expected["answer"] = _coach_expected_tokens(expected["kind"])

    failures: list[str] = []
    for scorer_cfg in case["scorers"]:
        if scorer_cfg["type"] == "llm_judge":
            score = await _llm_judge_coach(case["input"]["text"], answer)
        else:
            score = apply_scorer(scorer_cfg, result, expected)
        if score < 1.0:
            failures.append(
                f"scorer={scorer_cfg['type']} field={scorer_cfg.get('field')} "
                f"got={answer!r} expected≈{expected.get('answer')}"
            )

    assert not failures, f"{case['id']}: {'; '.join(failures)}"
