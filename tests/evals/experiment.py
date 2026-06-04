"""Offline eval experiment — uploads YAML cases to LangFuse datasets and runs
them with scoring, producing dataset runs for the dashboard screenshots.

Why not pytest: the LangFuse OTLP exporter cleanup races pytest-asyncio's
per-test event-loop teardown (see ``test_runner``). Here a single long-lived
loop owns every trace, so scores land reliably.

Run (with ``just up`` live and LANGFUSE_* in .env):

    just eval-report                # all agents
    uv run python -m tests.evals.experiment categorization   # one agent

One dataset per agent folder (``ff-<agent>``), so the LangFuse UI shows the
per-agent split the diploma review asks for.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from langfuse import Evaluation

from family_finance.agents.categorizer import CATEGORIZER_SYSTEM, CategoryPrediction
from family_finance.agents.supervisor import route_after_supervisor, supervisor_node
from family_finance.domain import Category, Direction, Transaction, TransactionSource
from family_finance.infrastructure.llm import get_chat_model
from family_finance.infrastructure.observability.langfuse_setup import flush, get_langfuse
from family_finance.infrastructure.parsers import TinkoffCsvParser
from family_finance.infrastructure.security import check_injection
from tests.evals.scorers import apply_scorer

EVAL_FAMILY_ID = UUID("00000000-0000-0000-0000-000000000001")
EVAL_MEMBER_ID = UUID("00000000-0000-0000-0000-000000000002")
CASES_DIR = Path(__file__).parent / "cases"

TaskFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def _run_categorization(inp: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

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
        import_hash=f"exp-{inp['merchant_raw']}",
    )
    model = get_chat_model(tier="worker").with_structured_output(CategoryPrediction)
    prediction: CategoryPrediction = await model.ainvoke(
        [
            SystemMessage(content=CATEGORIZER_SYSTEM),
            HumanMessage(content=f"Продавец: {tx.merchant_raw}\nСумма: {tx.amount} ₽"),
        ],
    )
    return {"category": prediction.category.value, "confidence": prediction.confidence}


async def _run_csv(inp: dict[str, Any]) -> dict[str, Any]:
    transactions = TinkoffCsvParser().parse(
        inp["csv_content"].encode("utf-8"),
        family_id=EVAL_FAMILY_ID,
        member_id=EVAL_MEMBER_ID,
    )
    return {
        "count": len(transactions),
        "first_category": transactions[0].category.value if transactions else None,
        "has_transfer": any(tx.direction == Direction.TRANSFER for tx in transactions),
    }


async def _run_security(inp: dict[str, Any]) -> dict[str, Any]:
    verdict = await check_injection(inp["text"])
    return {"blocked": verdict.blocked}


async def _run_routing(inp: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    state: dict[str, Any] = {"messages": [HumanMessage(content=inp["text"])]}
    update = await supervisor_node(state)
    return {"route": route_after_supervisor({**state, **update})}


_TASKS: dict[str, TaskFn] = {
    "categorization": _run_categorization,
    "csv_parsing": _run_csv,
    "security": _run_security,
    "tool_routing": _run_routing,
}


def _load_cases(agent: str) -> list[dict[str, Any]]:
    paths = sorted((CASES_DIR / agent).glob("*.yaml"))
    return [yaml.safe_load(p.read_text()) for p in paths]


def _make_evaluator() -> Callable[..., list[Evaluation]]:
    def evaluator(
        *,
        output: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[Evaluation]:
        scorers = (metadata or {}).get("scorers", [])
        expected = expected_output or {}
        scores: list[Evaluation] = []
        passed = True
        for cfg in scorers:
            if cfg["type"] == "llm_judge":
                continue  # async LLM-judge runs only in the pytest gate (just eval)
            value = apply_scorer(cfg, output, expected)
            passed = passed and value >= 1.0
            scores.append(
                Evaluation(
                    name=f"{cfg['type']}:{cfg.get('field', '')}".rstrip(":"),
                    value=value,
                    comment=f"got={output.get(cfg.get('field'))}",
                )
            )
        scores.append(Evaluation(name="pass", value=1.0 if passed else 0.0))
        return scores

    return evaluator


async def _run_agent(agent: str) -> None:
    cases = _load_cases(agent)
    if not cases:
        print(f"[skip] no cases for {agent}")
        return

    lf = get_langfuse()
    dataset_name = f"ff-{agent}"
    lf.create_dataset(name=dataset_name)
    for case in cases:
        lf.create_dataset_item(
            dataset_name=dataset_name,
            id=f"{agent}-{case['id']}",
            input=case["input"],
            expected_output=case["expected"],
            metadata={"scorers": case["scorers"], "tags": case.get("tags", [])},
        )

    task = _TASKS[agent]

    async def _task(*, item: Any, **_: Any) -> dict[str, Any]:
        return await task(item.input)

    dataset = lf.get_dataset(dataset_name)
    result = dataset.run_experiment(
        name=f"{agent} eval",
        task=_task,
        evaluators=[_make_evaluator()],
        max_concurrency=4,
    )
    print(result.format())
    print(f"→ {result.dataset_run_url}\n")


async def _main(agents: list[str]) -> None:
    for agent in agents:
        await _run_agent(agent)
    flush()


if __name__ == "__main__":
    selected = sys.argv[1:] or list(_TASKS)
    asyncio.run(_main(selected))
