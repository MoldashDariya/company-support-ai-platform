"""Stream and static message presentation in Telegram."""

from __future__ import annotations

import logging
import time

from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from channels.telegram.formatting import prepare_telegram_message, streaming_preview_text
from channels.telegram.widgets import intent_actions_keyboard
from cognition.intent import classify_query_intent
from conversation.pipeline import SupportConversationPipeline
from domain.models import PipelineResult, UserInquiry
from runtime import settings

logger = logging.getLogger(__name__)


class TelegramPresenter:
    """Conversion-oriented Telegram presentation layer."""

    def __init__(self, pipeline: SupportConversationPipeline) -> None:
        self._pipeline = pipeline

    async def deliver(self, message: Message, inquiry: UserInquiry) -> None:
        chat_id = message.chat.id
        await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        intent = classify_query_intent(inquiry.text)

        try:
            if settings.ENABLE_STREAMING:
                await self._deliver_streaming(message, inquiry, intent)
            else:
                result = await self._pipeline.handle(inquiry)
                intent_name = result.query_intent or intent.name
                keyboard = intent_actions_keyboard(intent_name, inquiry.text)
                await self._send_result(message, result, keyboard, intent_name, inquiry.text)
        except Exception:
            logger.exception("Telegram delivery failed session=%s", inquiry.session_id)
            keyboard = intent_actions_keyboard(intent.name, inquiry.text)
            fallback = (
                f"Сейчас не могу обработать запрос. Позвоните: {settings.COMPANY_PHONE} "
                f"или напишите в WhatsApp — кнопки ниже."
            )
            await self._answer_html(message, fallback, keyboard)

    async def _deliver_streaming(
        self,
        message: Message,
        inquiry: UserInquiry,
        intent,
    ) -> None:
        placeholder = await message.answer("…")
        accumulated = ""
        last_edit = 0.0
        final_result: PipelineResult | None = None

        async for event in self._pipeline.handle_stream(inquiry):
            kind, payload = event
            if kind == "token":
                accumulated += payload
                now = time.monotonic()
                if now - last_edit >= settings.STREAM_EDIT_INTERVAL_SEC:
                    preview = streaming_preview_text(accumulated)
                    try:
                        await placeholder.edit_text(preview)
                    except TelegramBadRequest:
                        pass
                    last_edit = now
            elif kind == "done":
                final_result = payload

        if final_result is not None:
            intent_name = final_result.query_intent or intent.name
            keyboard = intent_actions_keyboard(intent_name, inquiry.text)
            await self._send_result(
                message,
                final_result,
                keyboard,
                intent_name,
                inquiry.text,
                placeholder,
            )

    async def _send_result(
        self,
        message: Message,
        result: PipelineResult,
        keyboard,
        intent_name: str,
        query: str,
        placeholder: Message | None = None,
    ) -> None:
        if result.is_first_contact:
            welcome = prepare_telegram_message(
                settings.WELCOME_MESSAGE,
                intent="company_overview",
                query="",
                add_nudge=True,
            )
            welcome_kb = intent_actions_keyboard("company_overview", "")
            await self._answer_html(message, welcome, welcome_kb)

        html_text = prepare_telegram_message(
            result.reply.text,
            intent=intent_name,
            query=query,
        )

        if placeholder is not None:
            try:
                await placeholder.edit_text(
                    html_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            except TelegramBadRequest:
                await self._answer_html(message, html_text, keyboard)
        else:
            await self._answer_html(message, html_text, keyboard)

        logger.info(
            "Telegram UX | intent=%s keyboard=%s msg_len=%d",
            intent_name,
            bool(keyboard),
            len(html_text),
        )

    async def _answer_html(
        self,
        message: Message,
        text: str,
        keyboard=None,
    ) -> None:
        await message.answer(
            text,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
