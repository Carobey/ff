"""Domain — pure Pydantic. Ноль зависимостей от LangGraph/aiogram/etc."""

from __future__ import annotations

from family_finance.domain.budget import Budget, BudgetStatus
from family_finance.domain.digest_schedule import DigestSchedule
from family_finance.domain.family import Family, FamilyMember
from family_finance.domain.receipt import Receipt, ReceiptItem
from family_finance.domain.savings_goal import GoalProgress, SavingsGoal
from family_finance.domain.subscription import Subscription
from family_finance.domain.transaction import Transaction
from family_finance.domain.types import (
    BankSource,
    Category,
    Currency,
    Direction,
    FamilyRole,
    MoneyAmount,
    TransactionSource,
    WalletPrivacy,
)

__all__ = [
    "BankSource",
    "Budget",
    "BudgetStatus",
    "Category",
    "Currency",
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
]
