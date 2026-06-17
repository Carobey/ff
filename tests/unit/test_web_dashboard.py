"""Unit tests for the local web dashboard data assembly."""

from __future__ import annotations

import json
import uuid
import zoneinfo
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from starlette.testclient import TestClient

from family_finance.application.ports import LedgerBucket, LedgerEntry
from family_finance.domain import (
    Budget,
    BudgetStatus,
    Category,
    Direction,
    Family,
    SavingsGoal,
    Subscription,
)
from family_finance.web import app as web_app
from family_finance.web.agent import AgentAnswer, _web_text
from family_finance.web.dashboard import DashboardFilters, build_dashboard, build_detail

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")


class FakeDashboardRepo:
    def __init__(self, *, family_id: uuid.UUID | None = None, empty: bool = False) -> None:
        self.family_id = family_id or uuid.uuid4()
        self.families = [] if empty else [Family(family_id=self.family_id, name="Семья")]
        self.goal = SavingsGoal(
            family_id=self.family_id,
            target_amount=Decimal("200000"),
            target_date=date(2026, 12, 31),
            created_at=datetime(2026, 3, 1, tzinfo=_MOSCOW),
        )

    async def list_families(self) -> list[Family]:
        return self.families

    async def query_aggregates(
        self,
        *,
        family_id: uuid.UUID,
        group_by: str,
        then_by: str | None = None,
        categories: Sequence[Category] = (),
        directions: Sequence[Direction] = (),
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[LedgerBucket]:
        del family_id, then_by, categories, end, limit
        direction = directions[0] if directions else None
        month = start.month if start is not None else 0
        if group_by == "total":
            totals = {
                (Direction.INCOME, 6): Decimal("100000"),
                (Direction.EXPENSE, 6): Decimal("40000"),
                (Direction.INCOME, 5): Decimal("80000"),
                (Direction.EXPENSE, 5): Decimal("50000"),
            }
            total = totals.get((direction, month), Decimal("0"))
            return [LedgerBucket(bucket="total", total=total, count=1)]
        if group_by == "day" and direction == Direction.INCOME:
            return [LedgerBucket(bucket="2026-06-05", total=Decimal("100000"), count=1)]
        if group_by == "day" and direction == Direction.EXPENSE:
            return [
                LedgerBucket(bucket="2026-06-02", total=Decimal("1200"), count=1),
                LedgerBucket(bucket="2026-06-09", total=Decimal("5000"), count=2),
            ]
        if group_by == "category" and direction == Direction.EXPENSE:
            return [
                LedgerBucket(
                    bucket=Category.FOOD_GROCERIES.value,
                    total=Decimal("18000"),
                    count=12,
                ),
                LedgerBucket(bucket=Category.TRANSPORT_TAXI.value, total=Decimal("7000"), count=4),
            ]
        if group_by == "merchant" and direction == Direction.EXPENSE:
            return [
                LedgerBucket(bucket="Пятёрочка", total=Decimal("18000"), count=12),
                LedgerBucket(bucket="Такси", total=Decimal("7000"), count=4),
            ]
        return []

    async def list_transactions(
        self,
        *,
        family_id: uuid.UUID,
        categories: Sequence[Category] = (),
        directions: Sequence[Direction] = (),
        start: datetime | None = None,
        end: datetime | None = None,
        order_by: str = "date_desc",
        limit: int = 20,
        merchant: str | None = None,
    ) -> list[LedgerEntry]:
        del family_id, categories, directions, start, end, order_by, limit, merchant
        return [
            LedgerEntry(
                occurred_at=datetime(2026, 6, 14, 10, tzinfo=UTC),
                amount=Decimal("1200"),
                direction=Direction.EXPENSE,
                category=Category.FOOD_GROCERIES,
                merchant="Пятёрочка",
            )
        ]

    async def get_budget_status(
        self,
        *,
        family_id: uuid.UUID,
        month_start: datetime,
        month_end: datetime,
    ) -> list[BudgetStatus]:
        del month_start, month_end
        return [
            BudgetStatus(
                budget=Budget(
                    family_id=family_id,
                    category=Category.FOOD_GROCERIES,
                    monthly_limit=Decimal("30000"),
                ),
                spent_this_month=Decimal("15000"),
            ),
            BudgetStatus(
                budget=Budget(
                    family_id=family_id,
                    category=Category.ENTERTAINMENT_SUBS,
                    monthly_limit=Decimal("5000"),
                ),
                spent_this_month=Decimal("6000"),
            ),
        ]

    async def get_primary_member_id(self, *, family_id: uuid.UUID) -> uuid.UUID | None:
        del family_id
        return uuid.uuid4()

    async def detect_recurring(self, *, family_id: uuid.UUID) -> list[Subscription]:
        del family_id
        return [
            Subscription(
                merchant="Netflix",
                category=Category.ENTERTAINMENT_SUBS,
                cadence_days=30,
                average_amount=Decimal("799"),
                last_amount=Decimal("799"),
                last_seen=datetime(2026, 6, 1, tzinfo=UTC),
                occurrences=6,
            )
        ]

    async def get_savings_goal(self, *, family_id: uuid.UUID) -> SavingsGoal | None:
        del family_id
        return self.goal

    async def net_cashflow(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> Decimal:
        del family_id, end
        if start == self.goal.created_at:
            return Decimal("50000")
        return Decimal("60000")

    async def category_breakdown(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
        direction: Direction = Direction.EXPENSE,
    ) -> list[tuple[Category, Decimal, int]]:
        del family_id, start, end, direction
        return [
            (Category.FOOD_GROCERIES, Decimal("18000"), 12),
            (Category.TRANSPORT_TAXI, Decimal("7000"), 4),
        ]


@pytest.mark.unit
async def test_build_dashboard_assembles_current_family_snapshot() -> None:
    family_id = uuid.uuid4()
    repo = FakeDashboardRepo(family_id=family_id)
    now = datetime(2026, 6, 15, 12, tzinfo=_MOSCOW)

    dashboard = await build_dashboard(repo=repo, now=now)

    assert dashboard.selected_family_id == str(family_id)
    assert dashboard.period_label == "Июнь 2026"
    assert dashboard.metrics[0].value == "100 000 ₽"
    assert dashboard.metrics[0].trend == "+25% к предыдущему периоду"
    assert dashboard.metrics[1].value == "40 000 ₽"
    assert dashboard.metrics[1].trend == "-20% к предыдущему периоду"
    assert dashboard.metrics[2].value == "+60 000 ₽"
    assert dashboard.metrics[3].value == "2 667 ₽"
    assert dashboard.categories[0].label == "Продукты"
    assert dashboard.categories[0].pct == 72
    assert dashboard.categories[0].bucket == Category.FOOD_GROCERIES.value
    assert dashboard.filters.period == "this_month"
    assert dashboard.filters.group_by == "category"
    assert dashboard.budgets[1].status == "over"
    assert dashboard.budgets[1].pct == 120
    assert dashboard.subscriptions_total == "799 ₽"
    assert dashboard.goal is not None
    assert dashboard.goal.pct == 25
    assert dashboard.goal.remaining == "150 000 ₽"
    assert dashboard.recent_transactions[0].amount == "-1 200 ₽"
    assert dashboard.daily_points[4].income == 100000

    json.dumps(dashboard.as_dict(), ensure_ascii=False)
    json.dumps(dashboard.chart_payload(), ensure_ascii=False)


@pytest.mark.unit
async def test_build_dashboard_returns_empty_state_without_families() -> None:
    repo = FakeDashboardRepo(empty=True)
    now = datetime(2026, 6, 15, 12, tzinfo=_MOSCOW)

    dashboard = await build_dashboard(repo=repo, now=now)

    assert dashboard.selected_family_id is None
    assert dashboard.empty_state is not None
    assert dashboard.metrics == []


@pytest.mark.unit
async def test_build_dashboard_supports_custom_breakdown_filter() -> None:
    repo = FakeDashboardRepo()
    now = datetime(2026, 6, 15, 12, tzinfo=_MOSCOW)

    dashboard = await build_dashboard(
        repo=repo,
        now=now,
        filters=DashboardFilters(
            period="custom",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 15),
            group_by="merchant",
            direction=Direction.EXPENSE,
        ),
    )

    assert dashboard.period_label == "01.06.2026 - 15.06.2026"
    assert dashboard.filters.period == "custom"
    assert dashboard.filters.group_by == "merchant"
    assert dashboard.categories[0].label == "Пятёрочка"
    assert dashboard.categories[0].bucket == "Пятёрочка"


@pytest.mark.unit
async def test_build_detail_returns_transactions_for_bucket() -> None:
    family_id = uuid.uuid4()
    repo = FakeDashboardRepo(family_id=family_id)
    now = datetime(2026, 6, 15, 12, tzinfo=_MOSCOW)

    detail = await build_detail(
        family_id,
        repo=repo,
        now=now,
        filters=DashboardFilters(group_by="category"),
        bucket=Category.FOOD_GROCERIES.value,
    )

    assert detail.title == "Продукты"
    assert detail.transactions[0].merchant == "Пятёрочка"


# ── HTTP layer (app.py) ───────────────────────────────────────────────────────

_FIXED_NOW = datetime(2026, 6, 15, 12, tzinfo=_MOSCOW)


def _patch_build_dashboard(monkeypatch: pytest.MonkeyPatch, repo: FakeDashboardRepo) -> None:
    """Route the app's build_dashboard through a fake repo with a fixed clock."""
    real = web_app.build_dashboard

    async def fake(family_id: uuid.UUID | None = None, *, filters: object = None) -> object:
        return await real(family_id=family_id, filters=filters, repo=repo, now=_FIXED_NOW)  # type: ignore[arg-type]

    monkeypatch.setattr(web_app, "build_dashboard", fake)


@pytest.mark.unit
def test_dashboard_page_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_dashboard(monkeypatch, FakeDashboardRepo())
    client = TestClient(web_app.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Финансовая панель" in response.text


@pytest.mark.unit
def test_dashboard_page_bad_family_id_is_400() -> None:
    """A malformed family_id is a bad request, not a 'family not found' page."""
    client = TestClient(web_app.app)

    response = client.get("/", params={"family_id": "not-a-uuid"})

    assert response.status_code == 400


@pytest.mark.unit
def test_dashboard_page_bad_date_is_400() -> None:
    """An unparsable custom date is a 400, not a 404/503."""
    client = TestClient(web_app.app)

    response = client.get("/", params={"period": "custom", "start_date": "2026-13-40"})

    assert response.status_code == 400


@pytest.mark.unit
def test_dashboard_api_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_dashboard(monkeypatch, FakeDashboardRepo())
    client = TestClient(web_app.app)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    assert response.json()["metrics"]


@pytest.mark.unit
def test_dashboard_api_503_hides_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infra failure must not echo internal details (DSN, traceback) to the client."""

    async def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("postgres dsn host=secret-db password=hunter2")

    monkeypatch.setattr(web_app, "build_dashboard", boom)
    client = TestClient(web_app.app, raise_server_exceptions=False)

    response = client.get("/api/dashboard")

    assert response.status_code == 503
    assert response.json()["error"] == web_app._SERVICE_UNAVAILABLE
    assert "secret-db" not in response.text


@pytest.mark.unit
def test_agent_api_requires_family_id() -> None:
    client = TestClient(web_app.app)

    response = client.post("/api/agent", json={"question": "привет"})

    assert response.status_code == 400


@pytest.mark.unit
def test_agent_api_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ask(*, family_id: uuid.UUID, question: str) -> AgentAnswer:
        del family_id, question
        return AgentAnswer(answer="расходы под контролем")

    monkeypatch.setattr(web_app, "ask_agent", fake_ask)
    client = TestClient(web_app.app)

    response = client.post(
        "/api/agent",
        json={"family_id": str(uuid.uuid4()), "question": "как дела?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "расходы под контролем"


@pytest.mark.unit
def test_web_text_strips_telegram_tags() -> None:
    assert _web_text("<b>Привет</b>, <i>мир</i>") == "Привет, мир"
