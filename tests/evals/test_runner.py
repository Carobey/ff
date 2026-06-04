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
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from family_finance.agents.categorizer import CATEGORIZER_SYSTEM, CategoryPrediction
from family_finance.agents.supervisor import route_after_supervisor, supervisor_node
from family_finance.domain import Category, Direction, Transaction, TransactionSource
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.parsers import TinkoffCsvParser
from family_finance.infrastructure.security import check_injection
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


# ── Constants ─────────────────────────────────────────────────────────────────

EVAL_FAMILY_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
EVAL_MEMBER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
CASES_DIR = Path(__file__).parent / "cases"


def _load_cases(agent: str) -> list[Path]:
    return sorted((CASES_DIR / agent).glob("*.yaml"))


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
    prediction: CategoryPrediction = await model.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(content=CATEGORIZER_SYSTEM),
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
