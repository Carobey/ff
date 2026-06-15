"""aiogram handlers."""

from aiogram import Dispatcher

from family_finance.bot.handlers.clarify_buttons import router as clarify_buttons_router
from family_finance.bot.handlers.documents import router as documents_router
from family_finance.bot.handlers.import_confirm_buttons import (
    router as import_confirm_buttons_router,
)
from family_finance.bot.handlers.photos import router as photos_router
from family_finance.bot.handlers.start import router as start_router


def register_all_handlers(dp: Dispatcher) -> None:
    """Подключить все routers к dispatcher."""
    # Важно: callback_query handler должен быть зарегистрирован ДО message handlers
    dp.include_router(clarify_buttons_router)
    dp.include_router(import_confirm_buttons_router)
    dp.include_router(documents_router)
    dp.include_router(photos_router)
    dp.include_router(start_router)


__all__ = ["register_all_handlers"]
