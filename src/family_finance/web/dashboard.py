"""Read-only data assembly for the local finance dashboard."""

from __future__ import annotations

import uuid
import zoneinfo
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Protocol

from family_finance.agents.budgets import current_moscow_month
from family_finance.application.ports import LedgerBucket, LedgerEntry
from family_finance.domain import (
    BudgetStatus,
    Category,
    Direction,
    Family,
    GoalProgress,
    SavingsGoal,
    Subscription,
)
from family_finance.infrastructure.persistence import PostgresTransactionRepository

_MOSCOW = zoneinfo.ZoneInfo("Europe/Moscow")
_DAYS_PER_MONTH = Decimal("30")

_MONTH_NAMES = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь",
}

_CATEGORY_LABELS: dict[Category, str] = {
    Category.FOOD_GROCERIES: "Продукты",
    Category.FOOD_RESTAURANT: "Рестораны",
    Category.FOOD_DELIVERY: "Доставка еды",
    Category.FOOD_COFFEE: "Кофе",
    Category.TRANSPORT_FUEL: "Топливо",
    Category.TRANSPORT_TAXI: "Такси",
    Category.TRANSPORT_CARSHARE: "Каршеринг",
    Category.TRANSPORT_PUBLIC: "Общественный транспорт",
    Category.TRANSPORT_CARPARTS: "Автотовары",
    Category.KIDS_CLOTHES: "Детская одежда",
    Category.KIDS_TOYS: "Детские товары",
    Category.KIDS_SCHOOL: "Школа",
    Category.KIDS_ACTIVITIES: "Детские занятия",
    Category.SHOPPING_CLOTHES: "Одежда",
    Category.SHOPPING_GENERIC: "Покупки",
    Category.HOME_UTILITIES: "ЖКХ",
    Category.HOME_TELECOM: "Связь и интернет",
    Category.HOME_RENT: "Аренда",
    Category.HOME_FURNITURE: "Мебель",
    Category.HOME_REPAIR: "Ремонт",
    Category.HOME_HOUSEHOLD: "Дом",
    Category.HEALTH_PHARMACY: "Аптеки",
    Category.HEALTH_GENERIC: "Здоровье",
    Category.HEALTH_FITNESS: "Фитнес",
    Category.ENTERTAINMENT_SUBS: "Подписки",
    Category.ENTERTAINMENT_EVENTS: "События",
    Category.ENTERTAINMENT_HOBBIES: "Хобби",
    Category.ENTERTAINMENT_GAMES: "Игры",
    Category.BEAUTY_CARE: "Красота",
    Category.TRAVEL_TICKETS: "Билеты",
    Category.TRAVEL_LODGING: "Отели",
    Category.EDUCATION_COURSES: "Обучение",
    Category.FINANCE_FEES: "Комиссии",
    Category.FINANCE_LOAN: "Кредиты",
    Category.FINANCE_INSURANCE: "Страхование",
    Category.FINANCE_CASH: "Наличные",
    Category.FINANCE_INVESTMENT: "Инвестиции",
    Category.GOVERNMENT_FEES: "Госплатежи",
    Category.GIFTS: "Подарки",
    Category.CHARITY: "Благотворительность",
    Category.PETS: "Питомцы",
    Category.TAX_DED_MEDICAL: "Медицина",
    Category.TAX_DED_EDUCATION: "Образование",
    Category.TAX_DED_SPORT: "Спорт",
    Category.TAX_DED_IIS: "ИИС",
    Category.TAX_DED_PROPERTY: "Имущество",
    Category.INCOME_SALARY: "Зарплата",
    Category.INCOME_OTHER: "Доходы",
    Category.TRANSFER_INTERNAL: "Переводы",
    Category.UNCLASSIFIED: "Без категории",
}

_DIRECTION_LABELS: dict[Direction, str] = {
    Direction.EXPENSE: "Расход",
    Direction.INCOME: "Доход",
    Direction.TRANSFER: "Перевод",
    Direction.REFUND: "Возврат",
}

_PERIOD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("this_month", "Текущий месяц"),
    ("prev_month", "Прошлый месяц"),
    ("last_30", "30 дней"),
    ("last_90", "90 дней"),
    ("year", "Год"),
    ("custom", "Свой период"),
)

_GROUP_OPTIONS: tuple[tuple[str, str], ...] = (
    ("category", "Категории"),
    ("merchant", "Продавцы"),
    ("day", "Дни"),
    ("week", "Недели"),
    ("month", "Месяцы"),
)


class DashboardRepository(Protocol):
    """Read shape required by the dashboard service."""

    async def list_families(self) -> list[Family]: ...

    async def get_primary_member_id(self, *, family_id: uuid.UUID) -> uuid.UUID | None: ...

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
    ) -> list[LedgerBucket]: ...

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
    ) -> list[LedgerEntry]: ...

    async def get_budget_status(
        self,
        *,
        family_id: uuid.UUID,
        month_start: datetime,
        month_end: datetime,
    ) -> list[BudgetStatus]: ...

    async def detect_recurring(self, *, family_id: uuid.UUID) -> list[Subscription]: ...

    async def get_savings_goal(self, *, family_id: uuid.UUID) -> SavingsGoal | None: ...

    async def net_cashflow(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> Decimal: ...

    async def category_breakdown(
        self,
        *,
        family_id: uuid.UUID,
        start: datetime,
        end: datetime,
        direction: Direction = Direction.EXPENSE,
    ) -> list[tuple[Category, Decimal, int]]: ...


@dataclass(frozen=True)
class FamilyOption:
    family_id: str
    name: str


@dataclass(frozen=True)
class FilterOption:
    value: str
    label: str


@dataclass(frozen=True)
class DashboardFilters:
    period: str = "this_month"
    start_date: date | None = None
    end_date: date | None = None
    category: Category | None = None
    direction: Direction = Direction.EXPENSE
    group_by: str = "category"


@dataclass(frozen=True)
class FilterView:
    period: str
    start_date: str
    end_date: str
    category: str
    direction: str
    group_by: str


@dataclass(frozen=True)
class ResolvedDashboardFilters:
    start: datetime
    end: datetime
    period_label: str
    view: FilterView


@dataclass(frozen=True)
class MetricView:
    label: str
    value: str
    detail: str
    trend: str
    tone: str


@dataclass(frozen=True)
class BreakdownItemView:
    label: str
    bucket: str
    group_by: str
    total: str
    count: int
    pct: int
    bar_pct: int


@dataclass(frozen=True)
class BudgetView:
    label: str
    category: str
    spent: str
    limit: str
    pct: int
    bar_pct: int
    status: str
    status_label: str


@dataclass(frozen=True)
class SubscriptionView:
    merchant: str
    category: str
    monthly: str
    last_amount: str
    cadence: str
    last_seen: str
    occurrences: int


@dataclass(frozen=True)
class GoalView:
    target: str
    saved: str
    remaining: str
    pct: int
    bar_pct: int
    target_date: str
    monthly_needed: str | None
    status: str
    status_label: str


@dataclass(frozen=True)
class TransactionView:
    occurred_at: str
    merchant: str
    category: str
    amount: str
    direction: str
    tone: str


@dataclass(frozen=True)
class ChartPointView:
    label: str
    income: int
    expense: int


@dataclass(frozen=True)
class DashboardView:
    generated_at: str
    period_label: str
    families: list[FamilyOption]
    selected_family_id: str | None
    selected_family_name: str | None
    empty_state: str | None
    filters: FilterView
    period_options: list[FilterOption]
    category_options: list[FilterOption]
    direction_options: list[FilterOption]
    group_options: list[FilterOption]
    metrics: list[MetricView]
    categories: list[BreakdownItemView]
    budgets: list[BudgetView]
    subscriptions: list[SubscriptionView]
    subscriptions_total: str
    goal: GoalView | None
    recent_transactions: list[TransactionView]
    daily_points: list[ChartPointView]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def chart_payload(self) -> dict[str, object]:
        """Compact payload consumed by the tiny front-end chart script."""
        return {
            "daily": [asdict(point) for point in self.daily_points],
            "categories": [asdict(item) for item in self.categories[:8]],
        }


@dataclass(frozen=True)
class DetailView:
    title: str
    transactions: list[TransactionView]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


async def build_dashboard(
    family_id: uuid.UUID | None = None,
    *,
    filters: DashboardFilters | None = None,
    repo: DashboardRepository | None = None,
    now: datetime | None = None,
) -> DashboardView:
    """Build the current dashboard snapshot for one family."""
    resolved_now = _as_moscow(now or datetime.now(_MOSCOW))
    resolved_filters = resolve_dashboard_filters(filters or DashboardFilters(), now=resolved_now)
    dashboard_repo: DashboardRepository = repo or PostgresTransactionRepository()
    families = await dashboard_repo.list_families()
    family_options = [FamilyOption(family_id=str(f.family_id), name=f.name) for f in families]

    selected_family = _select_family(families, family_id)
    if selected_family is None:
        return _empty_dashboard(
            now=resolved_now,
            families=family_options,
            filters=resolved_filters,
            empty_state="В базе пока нет семей. Импортируй выписку через Telegram-бота.",
        )

    period_start = resolved_filters.start
    period_end = resolved_filters.end
    prev_start, prev_end = _previous_period(period_start, period_end)

    current_income = await _total(
        dashboard_repo,
        family_id=selected_family.family_id,
        directions=(Direction.INCOME,),
        start=period_start,
        end=period_end,
    )
    current_expense = await _total(
        dashboard_repo,
        family_id=selected_family.family_id,
        directions=(Direction.EXPENSE,),
        start=period_start,
        end=period_end,
    )
    previous_income = await _total(
        dashboard_repo,
        family_id=selected_family.family_id,
        directions=(Direction.INCOME,),
        start=prev_start,
        end=prev_end,
    )
    previous_expense = await _total(
        dashboard_repo,
        family_id=selected_family.family_id,
        directions=(Direction.EXPENSE,),
        start=prev_start,
        end=prev_end,
    )
    net = await dashboard_repo.net_cashflow(
        family_id=selected_family.family_id,
        start=period_start,
        end=period_end,
    )

    breakdown_raw = await _breakdown_rows(
        dashboard_repo,
        family_id=selected_family.family_id,
        filters=filters or DashboardFilters(),
        resolved=resolved_filters,
    )
    budgets_raw = await dashboard_repo.get_budget_status(
        family_id=selected_family.family_id,
        month_start=period_start,
        month_end=period_end,
    )
    subscriptions_raw = await dashboard_repo.detect_recurring(family_id=selected_family.family_id)
    goal_raw = await dashboard_repo.get_savings_goal(family_id=selected_family.family_id)
    recent_raw = await dashboard_repo.list_transactions(
        family_id=selected_family.family_id,
        order_by="date_desc",
        limit=10,
    )
    daily_points = await _daily_points(
        dashboard_repo,
        family_id=selected_family.family_id,
        month_start=period_start,
        month_end=period_end,
        now=resolved_now,
    )

    subscriptions_total = sum(
        (_subscription_monthly_amount(sub) for sub in subscriptions_raw),
        Decimal("0"),
    )

    return DashboardView(
        generated_at=resolved_now.strftime("%d.%m.%Y %H:%M"),
        period_label=resolved_filters.period_label,
        families=family_options,
        selected_family_id=str(selected_family.family_id),
        selected_family_name=selected_family.name,
        empty_state=None,
        filters=resolved_filters.view,
        period_options=_period_options(),
        category_options=_category_options(),
        direction_options=_direction_options(),
        group_options=_group_options(),
        metrics=[
            MetricView(
                label="Доходы",
                value=_money(current_income),
                detail="поступления за период",
                trend=_trend(current_income, previous_income),
                tone="income",
            ),
            MetricView(
                label="Расходы",
                value=_money(current_expense),
                detail="списания за период",
                trend=_trend(current_expense, previous_expense),
                tone="expense",
            ),
            MetricView(
                label="Чистый поток",
                value=_signed_money(net),
                detail="доходы и возвраты минус расходы",
                trend="за выбранный период",
                tone="positive" if net >= 0 else "negative",
            ),
            MetricView(
                label="Средний расход в день",
                value=_money(_daily_average(current_expense, period_start, resolved_now)),
                detail="по уже прошедшим дням периода",
                trend=f"{_elapsed_days(period_start, resolved_now)} дн.",
                tone="neutral",
            ),
        ],
        categories=_breakdown_views(
            breakdown_raw,
            total=sum((row.total for row in breakdown_raw), Decimal("0")),
            group_by=resolved_filters.view.group_by,
        ),
        budgets=_budget_views(budgets_raw),
        subscriptions=_subscription_views(subscriptions_raw),
        subscriptions_total=_money(subscriptions_total),
        goal=await _goal_view(
            dashboard_repo,
            family_id=selected_family.family_id,
            goal=goal_raw,
            now=resolved_now,
        ),
        recent_transactions=_transaction_views(recent_raw),
        daily_points=daily_points,
    )


async def build_detail(
    family_id: uuid.UUID,
    *,
    filters: DashboardFilters,
    bucket: str | None = None,
    repo: DashboardRepository | None = None,
    now: datetime | None = None,
) -> DetailView:
    """Return transaction rows for the selected dashboard bucket."""
    resolved_now = _as_moscow(now or datetime.now(_MOSCOW))
    resolved_filters = resolve_dashboard_filters(filters, now=resolved_now)
    dashboard_repo: DashboardRepository = repo or PostgresTransactionRepository()

    start = resolved_filters.start
    end = resolved_filters.end
    category = filters.category
    merchant: str | None = None
    title = "Операции"

    if bucket:
        if filters.group_by == "category":
            category = Category(bucket)
            title = _category_label(category)
        elif filters.group_by == "merchant":
            merchant = bucket
            title = bucket
        elif filters.group_by == "day":
            day = date.fromisoformat(bucket)
            start, end = _day_window(day)
            title = _bucket_label("day", bucket)
        elif filters.group_by == "week":
            week_start = date.fromisoformat(bucket)
            start, end = _date_window(week_start, week_start + timedelta(days=6))
            title = _bucket_label("week", bucket)
        elif filters.group_by == "month":
            year, month = bucket.split("-")
            start = datetime(int(year), int(month), 1, tzinfo=_MOSCOW)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
            title = _bucket_label("month", bucket)

    entries = await dashboard_repo.list_transactions(
        family_id=family_id,
        categories=(category,) if category else (),
        directions=(filters.direction,),
        start=start,
        end=end,
        order_by="amount_desc",
        limit=50,
        merchant=merchant,
    )
    return DetailView(title=title, transactions=_transaction_views(entries))


def resolve_dashboard_filters(
    filters: DashboardFilters,
    *,
    now: datetime,
) -> ResolvedDashboardFilters:
    """Resolve UI filter values to a concrete half-open datetime interval."""
    resolved_now = _as_moscow(now)
    period = filters.period if filters.period in {p for p, _ in _PERIOD_OPTIONS} else "this_month"
    today = resolved_now.date()

    if period == "custom":
        start_date = filters.start_date or today.replace(day=1)
        end_date = filters.end_date or today
        start, end = _date_window(start_date, end_date)
        label = f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
    elif period == "prev_month":
        current_start, _ = current_moscow_month(resolved_now)
        start, end = _previous_month(current_start)
        label = _period_label(start)
    elif period == "last_30":
        start, end = _date_window(today - timedelta(days=29), today)
        label = "Последние 30 дней"
    elif period == "last_90":
        start, end = _date_window(today - timedelta(days=89), today)
        label = "Последние 90 дней"
    elif period == "year":
        start = datetime(today.year, 1, 1, tzinfo=_MOSCOW)
        end = datetime.combine(today + timedelta(days=1), time.min, tzinfo=_MOSCOW)
        label = str(today.year)
    else:
        start, end = current_moscow_month(resolved_now)
        label = _period_label(start)

    return ResolvedDashboardFilters(
        start=start,
        end=end,
        period_label=label,
        view=FilterView(
            period=period,
            start_date=start.date().isoformat(),
            end_date=(end.date() - timedelta(days=1)).isoformat(),
            category=filters.category.value if filters.category else "",
            direction=filters.direction.value,
            group_by=filters.group_by if filters.group_by in _AGG_GROUPS else "category",
        ),
    )


_AGG_GROUPS = {"day", "week", "month", "category", "merchant"}


async def _breakdown_rows(
    repo: DashboardRepository,
    *,
    family_id: uuid.UUID,
    filters: DashboardFilters,
    resolved: ResolvedDashboardFilters,
) -> list[LedgerBucket]:
    group_by = resolved.view.group_by
    return await repo.query_aggregates(
        family_id=family_id,
        group_by=group_by,
        categories=(filters.category,) if filters.category else (),
        directions=(filters.direction,),
        start=resolved.start,
        end=resolved.end,
        limit=20,
    )


def _select_family(families: list[Family], family_id: uuid.UUID | None) -> Family | None:
    if not families:
        return None
    if family_id is None:
        return families[0]
    for family in families:
        if family.family_id == family_id:
            return family
    msg = f"family_id {family_id} not found"
    raise ValueError(msg)


def _empty_dashboard(
    *,
    now: datetime,
    families: list[FamilyOption],
    filters: ResolvedDashboardFilters,
    empty_state: str,
) -> DashboardView:
    return DashboardView(
        generated_at=now.strftime("%d.%m.%Y %H:%M"),
        period_label=filters.period_label,
        families=families,
        selected_family_id=None,
        selected_family_name=None,
        empty_state=empty_state,
        filters=filters.view,
        period_options=_period_options(),
        category_options=_category_options(),
        direction_options=_direction_options(),
        group_options=_group_options(),
        metrics=[],
        categories=[],
        budgets=[],
        subscriptions=[],
        subscriptions_total=_money(Decimal("0")),
        goal=None,
        recent_transactions=[],
        daily_points=[],
    )


async def _total(
    repo: DashboardRepository,
    *,
    family_id: uuid.UUID,
    directions: Sequence[Direction],
    start: datetime,
    end: datetime,
) -> Decimal:
    rows = await repo.query_aggregates(
        family_id=family_id,
        group_by="total",
        directions=directions,
        start=start,
        end=end,
        limit=1,
    )
    if not rows:
        return Decimal("0")
    return rows[0].total


async def _daily_points(
    repo: DashboardRepository,
    *,
    family_id: uuid.UUID,
    month_start: datetime,
    month_end: datetime,
    now: datetime,
) -> list[ChartPointView]:
    expense_rows = await repo.query_aggregates(
        family_id=family_id,
        group_by="day",
        directions=(Direction.EXPENSE,),
        start=month_start,
        end=month_end,
        limit=400,
    )
    income_rows = await repo.query_aggregates(
        family_id=family_id,
        group_by="day",
        directions=(Direction.INCOME,),
        start=month_start,
        end=month_end,
        limit=400,
    )
    expenses = {row.bucket: row.total for row in expense_rows}
    incomes = {row.bucket: row.total for row in income_rows}

    end_date = min(now.date(), (month_end - timedelta(days=1)).date())
    day = month_start.date()
    points: list[ChartPointView] = []
    while day <= end_date:
        key = day.isoformat()
        points.append(
            ChartPointView(
                label=str(day.day),
                income=_chart_amount(incomes.get(key, Decimal("0"))),
                expense=_chart_amount(expenses.get(key, Decimal("0"))),
            )
        )
        day += timedelta(days=1)
    return points


async def _goal_view(
    repo: DashboardRepository,
    *,
    family_id: uuid.UUID,
    goal: SavingsGoal | None,
    now: datetime,
) -> GoalView | None:
    if goal is None:
        return None

    saved = await repo.net_cashflow(family_id=family_id, start=goal.created_at, end=now)
    progress = GoalProgress(goal=goal, saved_so_far=saved)
    saved_display = saved if saved > 0 else Decimal("0")
    monthly_needed = progress.monthly_needed(now)
    on_track = progress.on_track(now)
    status, status_label = _goal_status(progress.reached, on_track)
    return GoalView(
        target=_money(goal.target_amount),
        saved=_money(saved_display),
        remaining=_money(progress.remaining),
        pct=progress.pct,
        bar_pct=_clamp_pct(progress.pct),
        target_date=goal.target_date.strftime("%d.%m.%Y") if goal.target_date else "без даты",
        monthly_needed=_money(monthly_needed) if monthly_needed is not None else None,
        status=status,
        status_label=status_label,
    )


def _breakdown_views(
    rows: list[LedgerBucket],
    *,
    total: Decimal,
    group_by: str,
) -> list[BreakdownItemView]:
    items: list[BreakdownItemView] = []
    for row in rows[:8]:
        pct = _percent(row.total, total)
        items.append(
            BreakdownItemView(
                label=_bucket_label(group_by, row.bucket),
                bucket=row.bucket,
                group_by=group_by,
                total=_money(row.total),
                count=row.count,
                pct=pct,
                bar_pct=_clamp_pct(pct),
            )
        )
    return items


def _budget_views(statuses: list[BudgetStatus]) -> list[BudgetView]:
    items: list[BudgetView] = []
    for status in statuses:
        state, label = _budget_status(status.pct)
        items.append(
            BudgetView(
                label=_category_label(status.budget.category),
                category=status.budget.category.value,
                spent=_money(status.spent_this_month),
                limit=_money(status.budget.monthly_limit),
                pct=status.pct,
                bar_pct=_clamp_pct(status.pct),
                status=state,
                status_label=label,
            )
        )
    return items


def _subscription_views(subscriptions: list[Subscription]) -> list[SubscriptionView]:
    return [
        SubscriptionView(
            merchant=sub.merchant,
            category=_category_label(sub.category),
            monthly=_money(_subscription_monthly_amount(sub)),
            last_amount=_money(sub.last_amount),
            cadence=f"~{sub.cadence_days} дн.",
            last_seen=sub.last_seen.astimezone(_MOSCOW).strftime("%d.%m.%Y"),
            occurrences=sub.occurrences,
        )
        for sub in subscriptions[:10]
    ]


def _transaction_views(entries: list[LedgerEntry]) -> list[TransactionView]:
    return [
        TransactionView(
            occurred_at=entry.occurred_at.astimezone(_MOSCOW).strftime("%d.%m.%Y"),
            merchant=entry.merchant or "(без продавца)",
            category=_category_label(entry.category),
            amount=_entry_amount(entry),
            direction=_DIRECTION_LABELS[entry.direction],
            tone=_direction_tone(entry.direction),
        )
        for entry in entries
    ]


def _subscription_monthly_amount(sub: Subscription) -> Decimal:
    return (sub.average_amount * _DAYS_PER_MONTH / Decimal(sub.cadence_days)).quantize(Decimal("1"))


def _budget_status(pct: int) -> tuple[str, str]:
    if pct >= 100:
        return "over", "превышен"
    if pct >= 80:
        return "warn", "близко к лимиту"
    return "ok", "в норме"


def _goal_status(reached: bool, on_track: bool | None) -> tuple[str, str]:
    if reached:
        return "reached", "цель достигнута"
    if on_track is True:
        return "ok", "по плану"
    if on_track is False:
        return "warn", "отстаём"
    return "neutral", "без дедлайна"


def _entry_amount(entry: LedgerEntry) -> str:
    if entry.direction == Direction.EXPENSE:
        return f"-{_money(entry.amount)}"
    if entry.direction in (Direction.INCOME, Direction.REFUND):
        return f"+{_money(entry.amount)}"
    return _money(entry.amount)


def _direction_tone(direction: Direction) -> str:
    if direction == Direction.EXPENSE:
        return "expense"
    if direction in (Direction.INCOME, Direction.REFUND):
        return "income"
    return "neutral"


def _trend(current: Decimal, previous: Decimal) -> str:
    if previous == Decimal("0"):
        if current == Decimal("0"):
            return "без движения"
        return "нет базы сравнения"
    pct = int(((current - previous) / previous * Decimal("100")).quantize(Decimal("1")))
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct}% к предыдущему периоду"


def _daily_average(total: Decimal, month_start: datetime, now: datetime) -> Decimal:
    return (total / Decimal(_elapsed_days(month_start, now))).quantize(Decimal("1"))


def _elapsed_days(month_start: datetime, now: datetime) -> int:
    return max(1, (now.date() - month_start.date()).days + 1)


def _percent(value: Decimal, total: Decimal) -> int:
    if total <= 0:
        return 0
    return int((value / total * Decimal("100")).quantize(Decimal("1")))


def _clamp_pct(value: int) -> int:
    return max(0, min(value, 100))


def _chart_amount(value: Decimal) -> int:
    return max(0, int(value.quantize(Decimal("1"))))


def _money(value: Decimal) -> str:
    rounded = int(value.copy_abs().quantize(Decimal("1")))
    sign = "-" if value < 0 else ""
    return f"{sign}{rounded:,}".replace(",", " ") + " ₽"


def _signed_money(value: Decimal) -> str:
    if value > 0:
        return f"+{_money(value)}"
    return _money(value)


def _category_label(category: Category) -> str:
    return _CATEGORY_LABELS.get(category, category.value)


def _bucket_label(group_by: str, bucket: str) -> str:
    if group_by == "category":
        try:
            return _category_label(Category(bucket))
        except ValueError:
            return bucket
    if group_by == "day":
        return date.fromisoformat(bucket).strftime("%d.%m.%Y")
    if group_by == "week":
        start = date.fromisoformat(bucket)
        end = start + timedelta(days=6)
        return f"{start.strftime('%d.%m')} - {end.strftime('%d.%m')}"
    if group_by == "month":
        year, month = bucket.split("-")
        return f"{_MONTH_NAMES[int(month)]} {year}"
    return bucket


def _period_options() -> list[FilterOption]:
    return [FilterOption(value=value, label=label) for value, label in _PERIOD_OPTIONS]


def _category_options() -> list[FilterOption]:
    options = [FilterOption(value="", label="Все категории")]
    options.extend(
        FilterOption(value=category.value, label=_category_label(category))
        for category in sorted(Category, key=lambda item: _category_label(item))
    )
    return options


def _direction_options() -> list[FilterOption]:
    return [
        FilterOption(value=direction.value, label=label)
        for direction, label in _DIRECTION_LABELS.items()
    ]


def _group_options() -> list[FilterOption]:
    return [FilterOption(value=value, label=label) for value, label in _GROUP_OPTIONS]


def _period_label(month_start: datetime) -> str:
    return f"{_MONTH_NAMES[month_start.month]} {month_start.year}"


def _previous_month(month_start: datetime) -> tuple[datetime, datetime]:
    if month_start.month == 1:
        start = month_start.replace(year=month_start.year - 1, month=12)
    else:
        start = month_start.replace(month=month_start.month - 1)
    return start, month_start


def _previous_period(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    delta = end - start
    return start - delta, start


def _date_window(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    start = datetime.combine(start_date, time.min, tzinfo=_MOSCOW)
    end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=_MOSCOW)
    return start, end


def _day_window(day: date) -> tuple[datetime, datetime]:
    return _date_window(day, day)


def _as_moscow(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=_MOSCOW)
    return value.astimezone(_MOSCOW)
