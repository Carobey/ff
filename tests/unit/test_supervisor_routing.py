"""Unit tests for deterministic supervisor routing."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from family_finance.agents import supervisor as supervisor_module
from family_finance.agents.state import SectionResult
from family_finance.agents.supervisor import (
    QueryPlan,
    route_after_ingest,
    route_after_supervisor,
    section_worker,
    supervisor_node,
    synthesizer_node,
)


@pytest.mark.unit
def test_route_csv_to_ingest() -> None:
    assert route_after_supervisor({"pending_csv": "uploads/test.csv"}) == "ingest"


@pytest.mark.unit
def test_route_after_ingest_with_rows_to_categorizer() -> None:
    assert route_after_ingest({"ingest_ok": True}) == "categorizer"


@pytest.mark.unit
def test_route_after_ingest_empty_to_end() -> None:
    assert route_after_ingest({"ingest_ok": False}) == "end"
    assert route_after_ingest({}) == "end"


@pytest.mark.unit
def test_route_query_to_ledger() -> None:
    assert route_after_supervisor({"current_intent": "query"}) == "ledger"


@pytest.mark.unit
def test_route_clarification_to_clarify() -> None:
    assert route_after_supervisor({"current_intent": "clarify"}) == "clarify"


@pytest.mark.unit
def test_route_idle_to_end() -> None:
    assert route_after_supervisor({"current_intent": "idle"}) == "end"


@pytest.mark.unit
def test_route_subscriptions_to_subscriptions() -> None:
    assert route_after_supervisor({"current_intent": "subscriptions"}) == "subscriptions"


@pytest.mark.unit
def test_route_tax_to_tax() -> None:
    assert route_after_supervisor({"current_intent": "tax"}) == "tax"


@pytest.mark.unit
async def test_supervisor_routes_tax_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """«налоговый вычет» → intent=tax по ключевому слову, без supervisor-LLM."""

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Tax routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="какой налоговый вычет я могу получить")]}
    )

    assert result["current_intent"] == "tax"


@pytest.mark.unit
async def test_supervisor_plans_tax_plus_spending_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    """«посчитай вычет, и сколько потратил» (запятая) → веер spending+tax по порядку."""

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Multi-intent planning must not instantiate the supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {
            "messages": [
                HumanMessage(content="посчитай налоговый вычет, и сколько потратил за апрель")
            ]
        }
    )

    assert result["current_intent"] == "multi"
    assert result["plan"] == ["spending", "tax"]


@pytest.mark.unit
async def test_supervisor_detects_subscriptions_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Subscriptions routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node({"messages": [HumanMessage(content="мои подписки")]})

    assert result["current_intent"] == "subscriptions"


@pytest.mark.unit
async def test_supervisor_routes_ledger_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Ledger routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node({"messages": [HumanMessage(content="сколько на аптеки?")]})

    assert result["current_intent"] == "query"


@pytest.mark.unit
async def test_supervisor_routes_clarification_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Clarification routing must not instantiate supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {
            "messages": [HumanMessage(content="1 одежда")],
            "open_questions": [
                {
                    "id": 1,
                    "reason": "непонятная категория",
                    "merchant_raw": "CONCEPT CLUB",
                    "payment_dates": ["2026-04-01"],
                    "count": 1,
                    "total": "4607.00",
                    "import_hashes": ["hash-1"],
                    "text": "Непонятная категория: 01.04.2026, «CONCEPT CLUB», 4607.00 ₽. Что это?",
                }
            ],
        }
    )

    assert result["current_intent"] == "clarify"


@pytest.mark.unit
async def test_supervisor_llm_planner_routes_single_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Промах ключевых слов → LLM-планировщик; один раздел → одиночный маршрут."""

    async def fake_plan(_text: str) -> QueryPlan:
        return QueryPlan(sections=["spending"])

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Single-section plan must not call the small-talk LLM")

    monkeypatch.setattr("family_finance.agents.supervisor._plan_query", fake_plan)
    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="а что там у меня с продуктовым магазином")]}
    )

    assert result["current_intent"] == "query"


@pytest.mark.unit
async def test_supervisor_llm_planner_fans_out_keywordless_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Бес-ключевой мульти-интент (нет запятых/ключевых слов) → веер через планировщик."""

    async def fake_plan(_text: str) -> QueryPlan:
        return QueryPlan(sections=["advice", "spending"])

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Multi-section plan must not call the small-talk LLM")

    monkeypatch.setattr("family_finance.agents.supervisor._plan_query", fake_plan)
    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="помоги мне понять мою ситуацию и что предпринять")]}
    )

    assert result["current_intent"] == "multi"
    # Планировщик вернул в обратном порядке — _ordered_sections нормализует.
    assert result["plan"] == ["spending", "advice"]
    assert result["section_results"] == []


@pytest.mark.unit
async def test_supervisor_llm_planner_routes_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    """Планировщик пометил pattern=true → coach, без small-talk LLM."""

    async def fake_plan(_text: str) -> QueryPlan:
        return QueryPlan(pattern=True)

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Pattern routing must not call the small-talk LLM")

    monkeypatch.setattr("family_finance.agents.supervisor._plan_query", fake_plan)
    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="а так ли часто я это делаю обычно")]}
    )

    assert result["current_intent"] == "pattern"


@pytest.mark.unit
async def test_supervisor_smalltalk_falls_through_to_direct_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой план (не про финансы) → прямой ответ LLM, intent=idle."""

    async def fake_plan(_text: str) -> QueryPlan:
        return QueryPlan()

    class _Resp:
        content = "Привет! Чем помочь?"

    class _Model:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr("family_finance.agents.supervisor._plan_query", fake_plan)
    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", lambda *a, **k: _Model())

    result = await supervisor_node({"messages": [HumanMessage(content="привет, как дела")]})

    assert result["current_intent"] == "idle"
    assert result["messages"]


# ── Orchestrator-worker: мульти-интент (ADR 0008) ────────────────────────────


@pytest.mark.unit
async def test_supervisor_plans_four_intent_query_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Запрос с 4 интентами и запятыми → план веера, общий период, без LLM."""

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Multi-intent planning must not instantiate the supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "проанализируй расходы за апрель, проверь бюджеты, "
                        "выяви подписки неучтённые, дай рекомендации на чём съэкономить"
                    )
                )
            ],
        }
    )

    assert result["current_intent"] == "multi"
    assert result["plan"] == ["spending", "budgets", "subscriptions", "advice"]
    assert str(result["period_start"]).startswith("2026-04-01")
    assert str(result["period_end"]).startswith("2026-05-01")
    # Накопитель секций сброшен под свежую партию.
    assert result["section_results"] == []


@pytest.mark.unit
async def test_supervisor_plans_spending_plus_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    """«проанализируй расходы …, дай рекомендации» → spending + advice (а не один ledger)."""

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Multi-intent planning must not instantiate the supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {
            "messages": [
                HumanMessage(content="проанализируй все расходы за апрель, дай рекомендации")
            ]
        }
    )

    assert result["current_intent"] == "multi"
    assert result["plan"] == ["spending", "advice"]


@pytest.mark.unit
async def test_single_clause_subscriptions_query_is_not_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """«сколько на подписки в апреле» — одна клауза (нет запятой) → одиночный ledger."""

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Single-clause query must route by keyword without LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="сколько на подписки в апреле")]}
    )

    assert result["current_intent"] == "query"
    assert "plan" not in result


@pytest.mark.unit
async def test_single_clause_two_families_takes_priority_not_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Одна клауза, но два интент-семейства (нет разделителя) → побеждает приоритет.

    «сколько потратил и стоит ли урезать траты» матчит и spending (сколько/потрат),
    и advice (урезать). Без запятой/точки-с-запятой это НЕ веер: ledger проверяется
    раньше advice в supervisor_node, поэтому уходит одиночным маршрутом в ledger,
    а вторичный интент (advice) отбрасывается. Веер требует разделитель клауз.
    """

    def fail_get_chat_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("Priority routing must not instantiate the supervisor LLM")

    monkeypatch.setattr("family_finance.agents.supervisor.get_chat_model", fail_get_chat_model)

    result = await supervisor_node(
        {"messages": [HumanMessage(content="сколько потратил и стоит ли урезать траты")]}
    )

    assert result["current_intent"] == "query"  # ledger > advice по приоритету
    assert "plan" not in result  # вторичный интент отброшен, веера нет


@pytest.mark.unit
def test_route_after_supervisor_fans_out_on_multi() -> None:
    """current_intent='multi' → по одному Send('section_worker') на секцию плана."""
    family_id = str(uuid.uuid4())
    sends = route_after_supervisor(
        {
            "current_intent": "multi",
            "plan": ["spending", "advice"],
            "family_id": family_id,
            "period_start": "2026-04-01T00:00:00+00:00",
            "period_end": "2026-05-01T00:00:00+00:00",
            "period_label": "за 04.2026",
        }
    )

    assert isinstance(sends, list)
    assert [s.node for s in sends] == ["section_worker", "section_worker"]
    kinds = [s.arg["kind"] for s in sends]
    assert kinds == ["spending", "advice"]
    assert all(s.arg["family_id"] == family_id for s in sends)
    assert all(s.arg["period_label"] == "за 04.2026" for s in sends)


@pytest.mark.unit
async def test_section_worker_dispatches_to_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """section_worker зовёт нужный builder и заворачивает результат в section_results."""
    section: SectionResult = {
        "kind": "spending",
        "order": 1,
        "title": "Расходы",
        "body": "Все расходы за 04.2026 — по категориям:\n• Продукты: 1 000 ₽\nИтого: 1 000 ₽",
    }

    async def fake_build_spending(*_args: object, **_kwargs: object) -> SectionResult:
        return section

    monkeypatch.setattr(supervisor_module, "build_spending_section", fake_build_spending)

    result = await section_worker(
        {
            "kind": "spending",
            "family_id": str(uuid.uuid4()),
            "period_start": "2026-04-01T00:00:00+00:00",
            "period_end": "2026-05-01T00:00:00+00:00",
            "period_label": "за 04.2026",
        }
    )

    assert result == {"section_results": [section]}


@pytest.mark.unit
async def test_synthesizer_orders_sections_and_joins() -> None:
    """Синтез сортирует секции по order (порядок Send не гарантирован) и склеивает."""
    advice: SectionResult = {"kind": "advice", "order": 4, "title": "Рекомендации", "body": "СОВЕТ"}
    spending: SectionResult = {
        "kind": "spending",
        "order": 1,
        "title": "Расходы",
        "body": "РАСХОДЫ",
    }

    result = await synthesizer_node({"section_results": [advice, spending]})

    content = str(result["messages"][0].content)
    assert content == "РАСХОДЫ\n\nСОВЕТ"
    assert result["current_intent"] == "idle"


@pytest.mark.unit
async def test_synthesizer_handles_empty_sections() -> None:
    """Нет секций → дружелюбный fallback, не падаем."""
    result = await synthesizer_node({"section_results": []})

    assert isinstance(result["messages"][0], AIMessage)
    assert result["current_intent"] == "idle"


@pytest.mark.unit
def test_accumulate_sections_resets_on_empty_batch() -> None:
    """Reducer: пустой апдейт сбрасывает партию, непустой — добавляет."""
    from family_finance.agents.state import accumulate_sections

    a: SectionResult = {"kind": "spending", "order": 1, "title": "Р", "body": "x"}
    b: SectionResult = {"kind": "advice", "order": 4, "title": "С", "body": "y"}

    assert accumulate_sections([a], []) == []  # planner reset
    assert accumulate_sections([a], [b]) == [a, b]  # worker fan-in append
    assert accumulate_sections(None, [a]) == [a]
