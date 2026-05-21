"""Composition root — wires platform modules into a runnable application."""

from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher

from channels.telegram.adapter import create_telegram_router
from channels.telegram.presenter import TelegramPresenter
from cognition.composer import GroundedResponseComposer
from cognition.engine import GroundedAnswerEngine
from cognition.provider import CerebrasLanguageModel
from memory.session_repository import SessionRepository
from conversation.pipeline import SupportConversationPipeline
from knowledge.hybrid_retriever import HybridRetriever
from runtime import settings
from safety.guards import InputGuard
from safety.throttle import SlidingWindowThrottle

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Install dependencies:\n"
    "  python3 -m venv .venv && source .venv/bin/activate\n"
    "  pip install -r requirements.txt"
)


def _llm_provider_name() -> str:
    base = settings.LLM_BASE_URL.lower()
    if "cerebras" in base:
        return "Cerebras"
    if "openai" in base:
        return "OpenAI"
    return "OpenAI-compatible LLM"


def validate_environment() -> None:
    """Validate required .env variables before starting heavy services."""
    missing: list[str] = []
    hints: list[str] = []

    if not settings.TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
        hints.append("  TELEGRAM_BOT_TOKEN — Telegram bot token from @BotFather")
    if not settings.LLM_API_KEY:
        missing.append("OPENAI_API_KEY or LLM_API_KEY")
        hints.append("  OPENAI_API_KEY — OpenAI API key (or LLM_API_KEY for other providers)")

    if missing:
        detail = "\n".join(hints)
        raise SystemExit(
            "Startup failed: missing required environment variables in .env:\n"
            f"  {', '.join(missing)}\n\n"
            f"Add the following to your .env file:\n{detail}\n\n"
            "See .env.example for a full template."
        )


class ApplicationContext:
    """Holds composed services for the support assistant runtime."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever.from_corpus(settings.CORPUS_PATH)

        self.memory = SessionRepository()
        self.guard = InputGuard()
        self.throttle = SlidingWindowThrottle()

        llm = CerebrasLanguageModel()
        composer = GroundedResponseComposer()
        engine = GroundedAnswerEngine(llm=llm, composer=composer)

        self.pipeline = SupportConversationPipeline(
            retriever=self.retriever,
            engine=engine,
            memory=self.memory,
            guard=self.guard,
            throttle=self.throttle,
        )
        self.presenter = TelegramPresenter(self.pipeline)
        self._llm = llm

    def log_provider_ready(self) -> None:
        provider = _llm_provider_name()
        logger.info(
            "%s provider initialized (model=%s)",
            provider,
            settings.LLM_MODEL,
        )

    async def initialize_telegram(self) -> Bot:
        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        me = await bot.get_me()
        username = me.username or "(no username)"
        logger.info("Telegram bot initialized (@%s)", username)
        return bot

    def build_dispatcher(self) -> Dispatcher:
        dp = Dispatcher()
        dp.include_router(create_telegram_router(self.presenter))
        return dp

    def build_bot(self) -> Bot:
        return Bot(token=settings.TELEGRAM_BOT_TOKEN)


def dependency_install_hint() -> str:
    return _INSTALL_HINT
