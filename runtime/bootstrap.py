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

    def validate_environment(self) -> None:
        missing = []
        if not settings.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not settings.LLM_API_KEY:
            missing.append("OPENAI_API_KEY (or LLM_API_KEY)")
        if missing:
            raise SystemExit(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See .env.example"
            )

    def build_dispatcher(self) -> Dispatcher:
        dp = Dispatcher()
        dp.include_router(create_telegram_router(self.presenter))
        return dp

    def build_bot(self) -> Bot:
        return Bot(token=settings.TELEGRAM_BOT_TOKEN)
