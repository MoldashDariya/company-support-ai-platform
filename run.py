"""Entry point for the Company Support AI Assistant platform."""

from __future__ import annotations

import asyncio
import logging

from runtime.bootstrap import ApplicationContext
from runtime.logging_config import configure_logging
from runtime import settings

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    ctx = ApplicationContext()
    ctx.validate_environment()

    logger.info(
        "Support platform ready | corpus_fragments=%d model=%s streaming=%s",
        ctx.retriever.fragment_count,
        settings.LLM_MODEL,
        settings.ENABLE_STREAMING,
    )

    bot = ctx.build_bot()
    dispatcher = ctx.build_dispatcher()
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
