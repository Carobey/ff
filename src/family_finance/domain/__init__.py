"""Domain — pure Pydantic. Ноль зависимостей от LangGraph/aiogram/etc."""

from __future__ import annotations

from family_finance.domain.budget import Budget, BudgetStatus
from family_finance.domain.digest_schedule import DigestSchedule
from family_finance.domain.family import Family, FamilyMember
from family_finance.domain.receipt import Receipt, ReceiptItem
from family_finance.domain.savings_goal import GoalProgress, SavingsGoal
from family_finance.domain.subscription import Subscription
from family_finance.domain.tax_deduction import (
    DeductionEstimate,
    DeductionInput,
    estimate_social_deductions,
)
from family_finance.domain.transaction import Transaction
from family_finance.domain.types import (
    SUBSCRIPTION_CATEGORIES,
    BankSource,
    Category,
    Currency,
    Direction,
    FamilyRole,
    MoneyAmount,
    TransactionSource,
    WalletPrivacy,
    direction_for_category,
    normalize_merchant,
)

__all__ = [
    "SUBSCRIPTION_CATEGORIES",
    "BankSource",
    "Budget",
    "BudgetStatus",
    "Category",
    "Currency",
    "DeductionEstimate",
    "DeductionInput",
    "DigestSchedule",
    "Direction",
    "Family",
    "FamilyMember",
    "FamilyRole",
    "GoalProgress",
    "MoneyAmount",
    "Receipt",
    "ReceiptItem",
    "SavingsGoal",
    "Subscription",
    "Transaction",
    "TransactionSource",
    "WalletPrivacy",
    "direction_for_category",
    "estimate_social_deductions",
    "normalize_merchant",
]
