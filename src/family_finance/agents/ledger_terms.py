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


# ВАЖНО: это НЕ дубль таблиц из ``budgets`` / ``clarifications`` — у них разная
# кардинальность и назначение, поэтому они намеренно живут отдельно:
#   • здесь (ledger_terms) — query-термин → НЕСКОЛЬКО категорий + направления, с
#     catch-all правилом. «Сколько на еду» должно охватить groceries+restaurant+
#     delivery, поэтому это 1→many fan-out для агрегаций ledger-ноды.
#   • ``budgets._RU_CATEGORY_ALIASES`` — бюджет-ввод → ОДНА категория (1→1),
#     причём «магазин» → FOOD_GROCERIES (продуктовый бюджет).
#   • ``clarifications._classify_answer`` — ответ-уточнение → ОДНА (категория,
#     направление), причём «магазин» → SHOPPING_GENERIC.
# Конфликт «магазин» (groceries vs shopping) показывает, что слить их в одну
# таблицу нельзя без изменения поведения — разделение осознанное.
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
        categories=(Category.HOME_TELECOM, Category.HOME_UTILITIES),
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
        # Точные стемы детских форм: «детск/детей/детям/детях/детьм».
        # НЕ голый «дет» — иначе «детальные/детально» ложно уходят в эту категорию.
        tokens=("детск", "детей", "детям", "детях", "детьм", "детса", "ребен", "школ", "секци"),
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
        # UNCLASSIFIED намеренно НЕ исключаем: «потратил/расходы» должны учитывать
        # и ещё не категоризированные траты, иначе суммы занижены (требование Юрия).
        categories=tuple(
            c
            for c in Category
            if c
            not in (
                Category.INCOME_SALARY,
                Category.INCOME_OTHER,
                Category.TRANSFER_INTERNAL,
            )
        ),
        directions=(Direction.EXPENSE,),
    ),
)

# Человекочитаемые подписи категорий для разбивки «по категориям» в ответах
# ledger-ноды. Неизвестный код деградирует до самого значения enum.
CATEGORY_LABELS: dict[Category, str] = {
    Category.FOOD_GROCERIES: "Продукты",
    Category.FOOD_RESTAURANT: "Рестораны",
    Category.FOOD_DELIVERY: "Доставка еды",
    Category.FOOD_COFFEE: "Кофе",
    Category.TRANSPORT_FUEL: "Топливо",
    Category.TRANSPORT_TAXI: "Такси",
    Category.TRANSPORT_CARSHARE: "Каршеринг",
    Category.TRANSPORT_PUBLIC: "Транспорт",
    Category.TRANSPORT_CARPARTS: "Автозапчасти",
    Category.KIDS_CLOTHES: "Детская одежда",
    Category.KIDS_TOYS: "Игрушки",
    Category.KIDS_SCHOOL: "Школа",
    Category.KIDS_ACTIVITIES: "Детские секции",
    Category.SHOPPING_CLOTHES: "Одежда",
    Category.SHOPPING_GENERIC: "Покупки",
    Category.HOME_UTILITIES: "ЖКХ",
    Category.HOME_TELECOM: "Связь",
    Category.HOME_RENT: "Аренда",
    Category.HOME_FURNITURE: "Мебель",
    Category.HOME_REPAIR: "Ремонт",
    Category.HOME_HOUSEHOLD: "Хозтовары",
    Category.HEALTH_PHARMACY: "Аптека",
    Category.HEALTH_GENERIC: "Здоровье",
    Category.HEALTH_FITNESS: "Фитнес",
    Category.ENTERTAINMENT_SUBS: "Подписки",
    Category.ENTERTAINMENT_EVENTS: "Мероприятия",
    Category.ENTERTAINMENT_HOBBIES: "Хобби",
    Category.ENTERTAINMENT_GAMES: "Игры",
    Category.BEAUTY_CARE: "Красота и уход",
    Category.TRAVEL_TICKETS: "Билеты",
    Category.TRAVEL_LODGING: "Жильё в поездках",
    Category.EDUCATION_COURSES: "Образование",
    Category.FINANCE_FEES: "Комиссии",
    Category.FINANCE_LOAN: "Кредиты",
    Category.FINANCE_INSURANCE: "Страховки",
    Category.FINANCE_CASH: "Наличные",
    Category.FINANCE_INVESTMENT: "Инвестиции",
    Category.GOVERNMENT_FEES: "Госплатежи",
    Category.GIFTS: "Подарки",
    Category.CHARITY: "Благотворительность",
    Category.PETS: "Питомцы",
    Category.TAX_DED_MEDICAL: "Медицина (вычет)",
    Category.TAX_DED_EDUCATION: "Образование (вычет)",
    Category.TAX_DED_SPORT: "Спорт (вычет)",
    Category.TAX_DED_IIS: "ИИС",
    Category.TAX_DED_PROPERTY: "Недвижимость (вычет)",
    Category.INCOME_SALARY: "Зарплата",
    Category.INCOME_OTHER: "Прочий доход",
    Category.TRANSFER_INTERNAL: "Перевод между своими",
    Category.UNCLASSIFIED: "Без категории",
}

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
