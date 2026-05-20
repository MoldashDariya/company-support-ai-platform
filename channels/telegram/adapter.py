"""Telegram channel adapter — routes messages to the support pipeline."""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message

from channels.telegram.presenter import TelegramPresenter
from channels.telegram.widgets import support_actions_keyboard
from domain.models import UserInquiry
from runtime import settings

logger = logging.getLogger(__name__)


def create_telegram_router(presenter: TelegramPresenter) -> Router:
    router = Router()

    @router.message(F.text)
    async def on_natural_message(message: Message) -> None:
        inquiry = UserInquiry(
            session_id=str(message.chat.id),
            text=message.text or "",
            channel="telegram",
        )
        await presenter.deliver(message, inquiry)

    @router.message()
    async def on_non_text(message: Message) -> None:
        await message.answer(
            f"Пожалуйста, отправьте текстовый вопрос о «{settings.COMPANY_NAME}».",
            reply_markup=support_actions_keyboard(),
        )

    return router
