"""Deterministic dictionaries for ledger query parsing."""

from __future__ import annotations

from dataclasses import dataclass

from family_finance.domain import Category, Direction


@dataclass(frozen=True)
class CategoryRule:
    """Maps user query words to canonical domain categories."""

    label: str
    tokens: tuple[str, ...]
    categories: tuple[Category, ...]
    directions: tuple[Direction, ...]


CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        label="супермаркеты",
        tokens=("супермаркет", "продукт", "пятероч"),
        categories=(Category.FOOD_GROCERIES,),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="еда",
        tokens=("еда", "еду", "еде"),
        categories=(Category.FOOD_GROCERIES, Category.FOOD_RESTAURANT, Category.FOOD_DELIVERY),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="аптеки",
        tokens=("аптек",),
        categories=(Category.HEALTH_PHARMACY,),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="заправки",
        tokens=("заправ", "топлив"),
        categories=(Category.TRANSPORT_FUEL,),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="авто",
        tokens=("авто", "машин"),
        categories=(Category.TRANSPORT_FUEL, Category.TRANSPORT_CARPARTS),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="ремонт и мебель",
        tokens=("ремонт", "мебел"),
        categories=(Category.HOME_REPAIR, Category.HOME_FURNITURE),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="связь",
        tokens=("связ", "мобил"),
        categories=(Category.HOME_UTILITIES,),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="переводы",
        tokens=("перевод",),
        categories=(Category.TRANSFER_INTERNAL,),
        directions=(Direction.TRANSFER,),
    ),
    CategoryRule(
        label="здоровье",
        tokens=("здоровь", "медицин", "клиник", "врач"),
        categories=(Category.HEALTH_PHARMACY, Category.HEALTH_GENERIC),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="развлечения",
        tokens=("развлечен", "подписк", "кино", "театр"),
        categories=(
            Category.ENTERTAINMENT_EVENTS,
            Category.ENTERTAINMENT_SUBS,
            Category.ENTERTAINMENT_HOBBIES,
        ),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="одежда",
        tokens=("одежд", "обув", "вайлдберр", "wildberr"),
        categories=(Category.SHOPPING_CLOTHES, Category.KIDS_CLOTHES),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="дом",
        tokens=("коммунал", "жкх", "квартир", "утилит"),
        categories=(Category.HOME_UTILITIES, Category.HOME_HOUSEHOLD),
        directions=(Direction.EXPENSE,),
    ),
    CategoryRule(
        label="дети",
        tokens=("дет", "ребен", "школ", "секци"),
        categories=(
            Category.KIDS_CLOTHES,
            Category.KIDS_TOYS,
            Category.KIDS_SCHOOL,
            Category.KIDS_ACTIVITIES,
        ),
        directions=(Direction.EXPENSE,),
    ),
    # Catch-all — must be last so specific rules have priority
    CategoryRule(
        label="все расходы",
        tokens=("потрат", "расход", "трат", "потратил", "трачу"),
        categories=tuple(
            c
            for c in Category
            if c
            not in (
                Category.INCOME_SALARY,
                Category.INCOME_OTHER,
                Category.TRANSFER_INTERNAL,
                Category.UNCLASSIFIED,
            )
        ),
        directions=(Direction.EXPENSE,),
    ),
)

MONTH_TOKENS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("январ",), 1),
    (("феврал",), 2),
    (("март",), 3),
    (("апрел",), 4),
    (("май", "мая", "мае"), 5),
    (("июн",), 6),
    (("июл",), 7),
    (("август",), 8),
    (("сентябр",), 9),
    (("октябр",), 10),
    (("ноябр",), 11),
    (("декабр",), 12),
)
