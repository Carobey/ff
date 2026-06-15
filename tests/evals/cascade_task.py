"""Deterministic eval task for the categorization cascade (Phase A).

No LLM: exercises the real ``PostgresMerchantRuleRepository`` against the live
DB — the «узнать продавца без LLM» path and the learning loop (ответ юзера →
правило семьи). Shared by the pytest gate (``test_runner``) and the LangFuse
dashboard generator (``experiment``) so both score the same logic.
"""

from __future__ import annotations

from typing import Any

from family_finance.domain import Category
from family_finance.infrastructure.persistence import (
    PostgresMerchantRuleRepository,
    PostgresTransactionRepository,
    loop_local_pool,
)
from family_finance.infrastructure.settings import get_settings

# Выделенная одноразовая семья для cascade-evals (FK для выученных правил).
_CASCADE_EVAL_TG_ID = 999_000_111


async def run_cascade(inp: dict[str, Any]) -> dict[str, Any]:
    """Прогнать одного продавца через каскад правил; вернуть исход для скореров.

    Если в кейсе есть ``learn_category`` — сначала имитируем ответ пользователя
    (upsert правила семьи), затем ищем: правило должно сработать (learning loop).
    """
    # The LangFuse eval runner drives each experiment on a throwaway event loop,
    # so the process-wide cached asyncpg pool would be bound to a dead loop. Bind
    # a fresh pool to the current loop for this item; the repos pick it up via the
    # shared cache (unchanged). The pytest gate runs on a live session loop and is
    # unaffected.
    async with loop_local_pool():
        family_id, _member_id = await PostgresTransactionRepository().ensure_member_for_telegram(
            telegram_user_id=_CASCADE_EVAL_TG_ID,
            name="Cascade eval family",
        )
        rule_repo = PostgresMerchantRuleRepository()
        merchant = inp["merchant_raw"]

        if "learn_category" in inp:
            await rule_repo.upsert(
                family_id=family_id,
                merchant_raw=merchant,
                category=Category(inp["learn_category"]),
                source="user",
            )

        hits = await rule_repo.lookup_many(
            family_id=family_id,
            merchants=[merchant],
            threshold=get_settings().merchant_match_threshold,
        )
        hit = hits.get(merchant)
        return {
            "rule_hit": hit is not None,
            "category": hit.category.value if hit is not None else None,
            "source": hit.source if hit is not None else None,
        }
