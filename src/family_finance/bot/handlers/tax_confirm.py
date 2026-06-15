"""HITL для налогового вычета (ADR 0010).

``tax_node`` ставит граф на паузу через ``interrupt(questions)`` перед точным
расчётом возврата НДФЛ: ему нужны годовой доход (если не виден в выписке) и флаги
«дорогостоящее лечение / обучение детей». Бот рендерит этот payload в текстовую
форму, а свободный ответ юзера парсит в ``dict`` и возобновляет граф через
``Command(resume=answers)``.

Парсинг fail-safe: что не распозналось — отсутствует в dict, ``tax_node``
трактует пропуски консервативно (доход из транзакций, флаги = 0). Никогда не
отвергаем ответ — пустой dict тоже валиден.
"""

from __future__ import annotations

import re
from typing import Any

# Денежные/числовые токены: «1 200 000», «50000» → одни цифры.
_NUMBER = r"(\d[\d  ]*)"


def format_tax_questions(payload: dict[str, Any]) -> str:
    """Render the ``interrupt()`` payload from ``tax_node`` into a text form."""
    year = payload.get("year")
    lines = [f"🧾 Чтобы точно посчитать возврат НДФЛ за {year} год, уточни:"]
    fields: list[str] = []
    if payload.get("need_income"):
        lines.append("• <b>Годовой доход</b> (₽) — зарплату в выписке не вижу.")
        fields.append("доход: 1200000")
    if payload.get("ask_medical_expensive"):
        total = payload.get("medical_total")
        lines.append(
            f"• Из медицины ({total} ₽) сколько было <b>дорогостоящего</b> "
            "лечения (оно без лимита)? Если не было — 0."
        )
        fields.append("дорогостоящее: 0")
    if payload.get("ask_children_education"):
        total = payload.get("education_total")
        lines.append(
            f"• Из обучения ({total} ₽) сколько за <b>детей</b> и сколько детей? "
            "Если за себя — 0."
        )
        fields.append("обучение детей: 0; число детей: 0")
    lines.append("")
    lines.append("Ответь одним сообщением, например:")
    lines.append("<code>" + "; ".join(fields) + "</code>")
    return "\n".join(lines)


def _find(text: str, alias: str) -> str | None:
    """Первое число после *alias* в тексте, нормализованное до одних цифр."""
    match = re.search(re.escape(alias) + r"\D*" + _NUMBER, text)
    if match is None:
        return None
    return re.sub(r"\D", "", match.group(1))


def parse_tax_answers(text: str) -> dict[str, str]:
    """Свободный текст юзера → ``dict`` для ``Command(resume=...)``.

    Ключи ищем по русским словам-меткам. «число детей» парсим раньше «обучение
    детей», чтобы счётчик детей не перехватил сумму. Что не нашли — не кладём.
    """
    normalized = text.lower().replace("\n", "; ")
    out: dict[str, str] = {}
    # (поле, метки в порядке убывания специфичности)
    for field, aliases in (
        ("annual_income", ("доход",)),
        ("medical_expensive", ("дорогост",)),
        ("children_count", ("число детей", "кол")),
        ("education_children", ("обучение детей", "дети", "детей")),
    ):
        for alias in aliases:
            value = _find(normalized, alias)
            if value:
                out[field] = value
                break
    return out
