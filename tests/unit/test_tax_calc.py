"""Чистый расчёт социальных вычетов (ст. 219 НК) — лимиты, прогрессия, корзины.

Числа считаются в Python (домен), без LLM и БД — поэтому тесты детерминированы
и фиксируют доменные правила 2025–2026: общий лимит 150k, обучение детей 110k×N,
дорогостоящее без лимита, благотворительность ≤25% дохода, ставка по прогрессии.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from family_finance.domain import DeductionInput, estimate_social_deductions
from family_finance.domain.tax_deduction import ndfl_marginal_rate


def _inp(**over: object) -> DeductionInput:
    base: dict[str, object] = {
        "annual_income": Decimal("1200000"),
        "medical_regular": Decimal("0"),
        "medical_expensive": Decimal("0"),
        "education_self": Decimal("0"),
        "education_children": Decimal("0"),
        "children_count": 0,
        "sport": Decimal("0"),
        "charity": Decimal("0"),
    }
    base.update(over)
    return DeductionInput(**base)  # type: ignore[arg-type]


# ── Прогрессивная шкала НДФЛ 2025 ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("income", "rate"),
    [
        ("0", "0.13"),
        ("2400000", "0.13"),  # граница 13% включительно
        ("2400001", "0.15"),
        ("5000000", "0.15"),
        ("5000001", "0.18"),
        ("20000000", "0.18"),
        ("20000001", "0.20"),
        ("50000000", "0.20"),
        ("50000001", "0.22"),
        ("99000000", "0.22"),
    ],
)
def test_ndfl_marginal_rate_brackets(income: str, rate: str) -> None:
    assert ndfl_marginal_rate(Decimal(income)) == Decimal(rate)


# ── Общий лимит 150k ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_general_bucket_under_limit() -> None:
    est = estimate_social_deductions(
        _inp(
            medical_regular=Decimal("50000"),
            education_self=Decimal("30000"),
            sport=Decimal("20000"),
        )
    )
    assert est.general_base == Decimal("100000")
    assert est.general_capped is False
    assert est.total_base == Decimal("100000")
    assert est.refund == Decimal("13000.00")  # 13% × 100000


@pytest.mark.unit
def test_general_bucket_caps_at_150k() -> None:
    est = estimate_social_deductions(
        _inp(
            medical_regular=Decimal("100000"),
            education_self=Decimal("60000"),
            sport=Decimal("20000"),  # сумма 180000 > 150000
        )
    )
    assert est.general_base == Decimal("150000")
    assert est.general_capped is True
    assert est.refund == Decimal("19500.00")  # 13% × 150000


# ── Дорогостоящее лечение — без лимита ─────────────────────────────────────────


@pytest.mark.unit
def test_expensive_medical_unlimited() -> None:
    est = estimate_social_deductions(
        _inp(
            annual_income=Decimal("2000000"),
            medical_expensive=Decimal("500000"),
        )
    )
    assert est.expensive_base == Decimal("500000")
    assert est.general_capped is False
    assert est.refund == Decimal("65000.00")  # 13% × 500000


# ── Обучение детей — отдельный лимит 110k на ребёнка ───────────────────────────


@pytest.mark.unit
def test_child_education_caps_per_child() -> None:
    est = estimate_social_deductions(
        _inp(
            education_children=Decimal("250000"),
            children_count=2,  # лимит 220000
        )
    )
    assert est.child_base == Decimal("220000")
    assert est.child_capped is True
    assert est.refund == Decimal("28600.00")  # 13% × 220000


@pytest.mark.unit
def test_child_education_no_children_means_zero_cap() -> None:
    est = estimate_social_deductions(_inp(education_children=Decimal("80000"), children_count=0))
    assert est.child_base == Decimal("0")
    assert est.child_capped is True
    assert est.refund == Decimal("0.00")


# ── Благотворительность — до 25% дохода ────────────────────────────────────────


@pytest.mark.unit
def test_charity_caps_at_quarter_of_income() -> None:
    est = estimate_social_deductions(
        _inp(
            annual_income=Decimal("300000"),
            charity=Decimal("100000"),  # 25% = 75000
        )
    )
    assert est.charity_base == Decimal("75000.00")
    assert est.charity_capped is True
    assert est.refund == Decimal("9750.00")  # 13% × 75000


# ── Сумма корзин + ставка по доходу ────────────────────────────────────────────


@pytest.mark.unit
def test_all_buckets_combined_with_higher_rate() -> None:
    est = estimate_social_deductions(
        _inp(
            annual_income=Decimal("3000000"),  # 15%
            medical_regular=Decimal("100000"),
            medical_expensive=Decimal("200000"),
            education_self=Decimal("100000"),  # general 200000 → cap 150000
            education_children=Decimal("90000"),
            children_count=1,  # cap 110000 → child_base 90000
            sport=Decimal("0"),
            charity=Decimal("0"),
        )
    )
    # general 150000 (capped) + expensive 200000 + child 90000 = 440000
    assert est.total_base == Decimal("440000")
    assert est.rate == Decimal("0.15")
    assert est.refund == Decimal("66000.00")  # 15% × 440000


@pytest.mark.unit
def test_refund_rounds_to_kopecks() -> None:
    est = estimate_social_deductions(_inp(medical_regular=Decimal("12345.67")))
    # 13% × 12345.67 = 1604.9371 → 1604.94
    assert est.refund == Decimal("1604.94")


# ── Валидация входа ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_negative_amount_rejected() -> None:
    with pytest.raises(ValidationError):
        _inp(medical_regular=Decimal("-1"))


@pytest.mark.unit
def test_amounts_coerced_via_str_not_float() -> None:
    est = estimate_social_deductions(_inp(medical_regular="50000", sport=20000))
    assert est.general_base == Decimal("70000")
