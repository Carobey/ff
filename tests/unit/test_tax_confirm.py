"""Tax-HITL bot rendering/parsing (ADR 0010, bot-слой).

``tax_node`` отдаёт ``interrupt(questions)``; бот рендерит форму и парсит свободный
ответ в ``dict`` для ``Command(resume=...)``. Проверяем обе чистые функции.
"""

from __future__ import annotations

from family_finance.bot.handlers.tax_confirm import (
    format_tax_questions,
    parse_tax_answers,
)


def _payload(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "tax_deduction_input",
        "year": 2025,
        "need_income": True,
        "ask_medical_expensive": True,
        "medical_total": "120000",
        "ask_children_education": True,
        "education_total": "90000",
    }
    base.update(over)
    return base


def test_format_lists_only_requested_fields() -> None:
    text = format_tax_questions(_payload(need_income=False, ask_children_education=False))
    assert "2025" in text
    assert "доход" not in text.lower()  # не спрашиваем — зарплата видна
    assert "дорогостоящего" in text
    assert "детей" not in text.lower()


def test_format_includes_example_line() -> None:
    text = format_tax_questions(_payload())
    assert "доход" in text.lower()
    assert "дорогостоящего" in text
    assert "<code>" in text  # пример формата ответа


def test_parse_full_answer() -> None:
    answers = parse_tax_answers(
        "доход: 1 200 000; дорогостоящее: 50000; обучение детей: 30000; число детей: 2"
    )
    assert answers == {
        "annual_income": "1200000",
        "medical_expensive": "50000",
        "education_children": "30000",
        "children_count": "2",
    }


def test_parse_count_does_not_steal_education_amount() -> None:
    # «число детей» должно матчиться раньше «детей», иначе счётчик съест сумму.
    answers = parse_tax_answers("обучение детей: 30000; число детей: 3")
    assert answers["children_count"] == "3"
    assert answers["education_children"] == "30000"


def test_parse_partial_keeps_only_found() -> None:
    answers = parse_tax_answers("доход 900000")
    assert answers == {"annual_income": "900000"}


def test_parse_garbage_is_empty_not_error() -> None:
    # Пустой dict валиден — tax_node достроит консервативно (fail-safe).
    assert parse_tax_answers("да, посчитай пожалуйста") == {}
