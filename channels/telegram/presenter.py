"""Stream and static message presentation in Telegram."""

from __future__ import annotations

import logging
import time

from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from channels.telegram.widgets import support_actions_keyboard
from conversation.pipeline import SupportConversationPipeline
from domain.models import PipelineResult, UserInquiry
from runtime import settings

logger = logging.getLogger(__name__)


class TelegramPresenter:
    """Thin presentation layer — formats pipeline output for Telegram UX."""

    def __init__(self, pipeline: SupportConversationPipeline) -> None:
        self._pipeline = pipeline

    async def deliver(self, message: Message, inquiry: UserInquiry) -> None:
        chat_id = message.chat.id
        await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        keyboard = support_actions_keyboard()

        try:
            if settings.ENABLE_STREAMING:
                await self._deliver_streaming(message, inquiry, keyboard)
            else:
                result = await self._pipeline.handle(inquiry)
                await self._send_result(message, result, keyboard)
        except Exception:
            logger.exception("Telegram delivery failed session=%s", inquiry.session_id)
            await message.answer(
                "Сейчас не могу обработать запрос. Попробуйте позже или "
                f"позвоните: {settings.COMPANY_PHONE}.",
                reply_markup=keyboard,
            )

    async def _deliver_streaming(
        self,
        message: Message,
        inquiry: UserInquiry,
        keyboard,
    ) -> None:
        placeholder = await message.answer("…", reply_markup=keyboard)
        accumulated = ""
        last_edit = 0.0
        final_result: PipelineResult | None = None

        async for event in self._pipeline.handle_stream(inquiry):
            kind, payload = event
            if kind == "token":
                accumulated += payload
                now = time.monotonic()
                if now - last_edit >= settings.STREAM_EDIT_INTERVAL_SEC:
                    preview = accumulated.strip() or "…"
                    try:
                        await placeholder.edit_text(preview)
                    except TelegramBadRequest:
                        pass
                    last_edit = now
            elif kind == "done":
                final_result = payload

        if final_result is not None:
            await self._send_result(message, final_result, keyboard, placeholder)

    async def _send_result(
        self,
        message: Message,
        result: PipelineResult,
        keyboard,
        placeholder: Message | None = None,
    ) -> None:
        if result.is_first_contact:
            await message.answer(settings.WELCOME_MESSAGE, reply_markup=keyboard)

        text = result.reply.text
        if placeholder is not None:
            try:
                await placeholder.edit_text(text, reply_markup=keyboard)
            except TelegramBadRequest:
                await message.answer(text, reply_markup=keyboard)
        else:
            await message.answer(text, reply_markup=keyboard)
