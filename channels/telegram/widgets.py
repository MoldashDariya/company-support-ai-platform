"""Telegram UI widgets — contact shortcuts, no command menus."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from runtime import settings


def support_actions_keyboard() -> InlineKeyboardMarkup:
    phone = settings.COMPANY_PHONE.replace(" ", "")
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📞 Позвонить", url=f"tel:{phone}"),
                InlineKeyboardButton(text="🌐 Сайт", url=settings.COMPANY_SITE),
            ],
        ]
    )
